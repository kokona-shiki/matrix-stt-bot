import os
import signal
import sys
import traceback
from typing import Union, Optional

import aiofiles
import asyncio
import uuid
import json

from nio import (
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
    JoinError,
    KeyVerificationCancel,
    KeyVerificationEvent,
    KeyVerificationKey,
    KeyVerificationMac,
    KeyVerificationStart,
    LocalProtocolError,
    LoginResponse,
    MatrixRoom,
    MegolmEvent,
    RoomMessageAudio,
    RoomEncryptedAudio,
    ToDeviceError,
    crypto,
    EncryptionError,
    WhoamiError,
    DownloadError,
)
from nio.store.database import SqliteStore

from faster_whisper import WhisperModel

from log import getlogger
from send_message import send_room_message


logger = getlogger()


class Bot:
    def __init__(
        self,
        homeserver: str,
        user_id: str,
        device_id: str,
        room_id: Union[str, None] = None,
        password: Union[str, None] = None,
        access_token: Union[str, None] = None,
        device_name: Union[str, None] = None,
        import_keys_path: Optional[str] = None,
        import_keys_password: Optional[str] = None,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: str = "models",
    ):
        if homeserver is None or user_id is None or device_id is None:
            logger.warning("homeserver, user_id and device_id are required")
            sys.exit(1)

        if password is None and access_token is None:
            logger.warning("password or access_token is required")
            sys.exit(1)

        self.homeserver = homeserver
        self.user_id = user_id
        self.password = password
        self.access_token = access_token
        self.device_name = (
            device_name if device_name is not None else "matrix-stt-bot"
        )
        self.device_id = device_id
        self.room_id = room_id
        self.import_keys_path = import_keys_path
        self.import_keys_password = import_keys_password
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root

        if self.model_size is None:
            self.model_size = "tiny"

        if self.device is None:
            self.device = "cpu"

        if self.compute_type is None:
            self.compute_type = "int8"

        if self.cpu_threads is None:
            self.cpu_threads = 0

        if self.num_workers is None:
            self.num_workers = 1

        if self.download_root is None:
            cwd = os.getcwd()
            self.download_root = os.path.join(cwd, "models")

            if not os.path.exists(self.download_root):
                os.mkdir(self.download_root)

        # ---------------------------------------------------------
        # Matrix client
        # ---------------------------------------------------------
        self.store_path = os.getcwd()

        self.config = AsyncClientConfig(
            store=SqliteStore,
            store_name="db",
            store_sync_tokens=True,
            encryption_enabled=True,
        )

        self.client = AsyncClient(
            homeserver=self.homeserver,
            user=self.user_id,
            device_id=self.device_id,
            config=self.config,
            store_path=self.store_path,
        )

        if self.access_token is not None:
            self.client.access_token = self.access_token

        # ---------------------------------------------------------
        # Event callbacks
        # ---------------------------------------------------------
        self.client.add_event_callback(
            self.message_callback,
            (
                RoomMessageAudio,
                RoomEncryptedAudio,
            ),
        )

        self.client.add_event_callback(
            self.decryption_failure,
            (MegolmEvent,),
        )

        self.client.add_event_callback(
            self.invite_callback,
            (InviteMemberEvent,),
        )

        self.client.add_to_device_callback(
            self.to_device_callback,
            (KeyVerificationEvent,),
        )

        # ---------------------------------------------------------
        # Whisper model
        # ---------------------------------------------------------
        logger.info("Loading Whisper model...")

        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
            download_root=self.download_root,
        )

        logger.info("Whisper model loaded")

        # ---------------------------------------------------------
        # Only allow one transcription at a time.
        #
        # This is important on CPU VPS:
        # multiple Matrix audio events can arrive simultaneously.
        # ---------------------------------------------------------
        self.transcribe_lock = asyncio.Lock()

        # ---------------------------------------------------------
        # Temporary audio directory
        # ---------------------------------------------------------
        self.output_dir = os.path.join(os.getcwd(), "output")

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    async def close(self, task: asyncio.Task = None) -> None:
        await self.client.close()

        if task is not None:
            task.cancel()

        logger.info("Bot closed!")

    # -------------------------------------------------------------
    # Matrix audio event callback
    # -------------------------------------------------------------
    async def message_callback(
        self,
        room: MatrixRoom,
        event: Union[RoomMessageAudio, RoomEncryptedAudio],
    ) -> None:
        if self.room_id is None:
            room_id = room.room_id
        else:
            if room.room_id != self.room_id:
                return

            room_id = self.room_id

        reply_to_event_id = event.event_id
        sender_id = event.sender

        if isinstance(event, (RoomMessageAudio, RoomEncryptedAudio)):
            try:
                asyncio.create_task(
                    self.main_function(
                        event,
                        room_id,
                        sender_id,
                        reply_to_event_id,
                    )
                )
            except Exception:
                logger.error(
                    "Failed to create audio processing task",
                    exc_info=True,
                )

    # -------------------------------------------------------------
    # Process audio message
    # -------------------------------------------------------------
    async def main_function(
        self,
        event: Union[RoomMessageAudio, RoomEncryptedAudio],
        room_id: str,
        sender_id: str,
        reply_to_event_id: str,
    ):
        media_type = None
        filename = None

        try:
            # -----------------------------------------------------
            # Construct temporary filename
            # -----------------------------------------------------
            ext = os.path.splitext(event.body)[-1]

            if not ext:
                ext = ".audio"

            filename = os.path.join(
                self.output_dir,
                str(uuid.uuid4()) + ext,
            )

            # -----------------------------------------------------
            # Download Matrix media
            # -----------------------------------------------------
            mxc = event.url

            resp = await self.download_mxc(mxc=mxc)

            if isinstance(resp, DownloadError):
                logger.error("Download of media file failed")
                return

            media_data = resp.body

            # -----------------------------------------------------
            # Unencrypted audio
            # -----------------------------------------------------
            if isinstance(event, RoomMessageAudio):
                media_type = resp.content_type

                async with aiofiles.open(filename, "wb") as f:
                    await f.write(media_data)

            # -----------------------------------------------------
            # Encrypted audio
            # -----------------------------------------------------
            elif isinstance(event, RoomEncryptedAudio):
                media_type = event.mimetype

                decrypted_data = crypto.attachments.decrypt_attachment(
                    media_data,
                    event.source["content"]["file"]["key"]["k"],
                    event.source["content"]["file"]["hashes"]["sha256"],
                    event.source["content"]["file"]["iv"],
                )

                async with aiofiles.open(filename, "wb") as f:
                    await f.write(decrypted_data)

            # -----------------------------------------------------
            # Validate audio type
            #
            # WhatsApp audio messages may be audio/ogg.
            # Matrix voice messages may use audio/mp4 with
            # filenames beginning with "recording".
            # -----------------------------------------------------
            evt_filename = event.source["content"].get(
                "filename",
                "",
            )

            is_supported_audio = (
                media_type == "audio/ogg"
                or (
                    media_type is not None
                    and media_type.startswith("audio/")
                    and evt_filename.startswith("recording")
                )
            )

            if not is_supported_audio:
                logger.info("Ignoring unsupported media type")
                return

            # -----------------------------------------------------
            # Transcribe
            #
            # Serialize Whisper inference so multiple incoming
            # messages don't run Whisper simultaneously on CPU.
            # -----------------------------------------------------
            await self.client.room_typing(room_id)

            async with self.transcribe_lock:
                message = await asyncio.to_thread(
                    self.transcribe,
                    filename,
                )

            # -----------------------------------------------------
            # No speech / empty result
            # -----------------------------------------------------
            if not message:
                logger.info("No speech detected")
                return

            # -----------------------------------------------------
            # Send transcription back to Matrix
            # -----------------------------------------------------
            await send_room_message(
                client=self.client,
                room_id=room_id,
                reply_message=message,
                sender_id=sender_id,
                reply_to_event_id=reply_to_event_id,
            )

        except Exception:
            logger.error(
                "Audio processing failed",
                exc_info=True,
            )

        finally:
            # -----------------------------------------------------
            # Always delete temporary audio file.
            #
            # Important for privacy.
            # -----------------------------------------------------
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                    logger.info("Temporary audio file removed")
                except Exception:
                    logger.error(
                        "Failed to remove temporary audio file",
                        exc_info=True,
                    )

    # -------------------------------------------------------------
    # Matrix decryption failure
    # -------------------------------------------------------------
    async def decryption_failure(
        self,
        room: MatrixRoom,
        event: MegolmEvent,
    ) -> None:
        if not isinstance(event, MegolmEvent):
            return

        # Do not log room ID, sender ID or event ID.
        logger.error(
            "Failed to decrypt Matrix message. "
            "Please make sure the bot session is verified."
        )

    # -------------------------------------------------------------
    # Matrix invite callback
    # -------------------------------------------------------------
    async def invite_callback(
        self,
        room: MatrixRoom,
        event: InviteMemberEvent,
    ) -> None:
        """Handle an incoming room invite."""

        logger.debug("Received room invite")

        # Attempt to join up to 3 times.
        for attempt in range(3):
            result = await self.client.join(room.room_id)

            if isinstance(result, JoinError):
                logger.error(
                    "Error joining room (attempt %d): %s",
                    attempt + 1,
                    result.message,
                )
            else:
                break
        else:
            logger.error("Unable to join room")
            return

        logger.info("Joined room")

    # -------------------------------------------------------------
    # Matrix device verification
    # -------------------------------------------------------------
    async def to_device_callback(
        self,
        event: KeyVerificationEvent,
    ) -> None:
        """Handle device verification events."""

        try:
            client = self.client

            logger.debug(
                f"Device verification event received: {type(event)}"
            )

            # -----------------------------------------------------
            # Verification start
            # -----------------------------------------------------
            if isinstance(event, KeyVerificationStart):

                if "emoji" not in event.short_authentication_string:
                    logger.info(
                        "Other device does not support emoji verification. "
                        "Aborting."
                    )
                    return

                resp = await client.accept_key_verification(
                    event.transaction_id
                )

                if isinstance(resp, ToDeviceError):
                    logger.error(
                        "accept_key_verification() failed"
                    )
                    return

                sas = client.key_verifications[
                    event.transaction_id
                ]

                todevice_msg = sas.share_key()

                resp = await client.to_device(todevice_msg)

                if isinstance(resp, ToDeviceError):
                    logger.error(
                        "to_device() failed during verification"
                    )

            # -----------------------------------------------------
            # Verification cancelled
            # -----------------------------------------------------
            elif isinstance(event, KeyVerificationCancel):

                logger.info(
                    "Device verification was cancelled"
                )

            # -----------------------------------------------------
            # Verification key
            # -----------------------------------------------------
            elif isinstance(event, KeyVerificationKey):

                sas = client.key_verifications[
                    event.transaction_id
                ]

                # Print emojis locally but don't log them.
                print(sas.get_emoji())

                # Automatic verification.
                yn = "y"

                if yn.lower() == "y":
                    logger.info(
                        "Device verification accepted"
                    )

                    resp = await client.confirm_short_auth_string(
                        event.transaction_id
                    )

                    if isinstance(resp, ToDeviceError):
                        logger.error(
                            "confirm_short_auth_string() failed"
                        )

                elif yn.lower() == "n":
                    logger.info(
                        "Device verification rejected"
                    )

                    resp = await client.cancel_key_verification(
                        event.transaction_id,
                        reject=True,
                    )

                    if isinstance(resp, ToDeviceError):
                        logger.error(
                            "cancel_key_verification() failed"
                        )

                else:
                    logger.info(
                        "Device verification cancelled"
                    )

                    resp = await client.cancel_key_verification(
                        event.transaction_id,
                        reject=False,
                    )

                    if isinstance(resp, ToDeviceError):
                        logger.error(
                            "cancel_key_verification() failed"
                        )

            # -----------------------------------------------------
            # Verification MAC
            # -----------------------------------------------------
            elif isinstance(event, KeyVerificationMac):

                sas = client.key_verifications[
                    event.transaction_id
                ]

                try:
                    todevice_msg = sas.get_mac()

                except LocalProtocolError:
                    logger.info(
                        "Device verification protocol was cancelled"
                    )

                else:
                    resp = await client.to_device(
                        todevice_msg
                    )

                    if isinstance(resp, ToDeviceError):
                        logger.error(
                            "to_device() failed during verification"
                        )
                        return

                    if sas.verified:
                        logger.info(
                            "Emoji verification was successful"
                        )
                    else:
                        logger.info(
                            "Emoji verification completed"
                        )

            else:
                logger.debug(
                    f"Received unexpected verification event: "
                    f"{type(event)}"
                )

        except BaseException:
            logger.error(
                "Device verification error",
                exc_info=True,
            )

    # -------------------------------------------------------------
    # Matrix login
    # -------------------------------------------------------------
    async def login(self) -> None:

        if self.access_token is not None:
            self.client.restore_login(
                user_id=self.user_id,
                device_id=self.device_id,
                access_token=self.access_token,
            )

            try:
                resp = await self.client.whoami()

            except Exception:
                await self.client.close()

                logger.error(
                    "Login check failed",
                    exc_info=True,
                )

                sys.exit(1)

            if isinstance(resp, WhoamiError):
                logger.error(
                    "Login failed, please check the access token"
                )
                sys.exit(1)

            logger.info(
                "Successfully logged in via access token"
            )

        else:
            try:
                resp = await self.client.login(
                    password=self.password,
                    device_name=self.device_name,
                )

                if not isinstance(resp, LoginResponse):
                    logger.error("Login failed")
                    print("Login Failed")
                    sys.exit(1)

                logger.info(
                    "Successfully logged in via password"
                )

            except Exception:
                logger.error(
                    "Login error",
                    exc_info=True,
                )

    # -------------------------------------------------------------
    # Matrix sync
    # -------------------------------------------------------------
    async def sync_forever(
        self,
        timeout=30000,
        full_state=True,
    ) -> None:
        await self.client.sync_forever(
            timeout=timeout,
            full_state=full_state,
        )

    # -------------------------------------------------------------
    # Download Matrix media
    #
    # Do not log MXC URL or response object for privacy.
    # -------------------------------------------------------------
    async def download_mxc(
        self,
        mxc: str,
        filename: Optional[str] = None,
    ):
        response = await self.client.download(
            mxc=mxc,
            filename=filename,
        )

        return response

    # -------------------------------------------------------------
    # Import encryption keys
    # -------------------------------------------------------------
    async def import_keys(self):
        resp = await self.client.import_keys(
            self.import_keys_path,
            self.import_keys_password,
        )

        if isinstance(resp, EncryptionError):
            logger.error(
                "import_keys failed"
            )
        else:
            logger.info(
                "import_keys succeeded"
            )

    # -------------------------------------------------------------
    # Whisper transcription
    # -------------------------------------------------------------
    def transcribe(self, filename: str) -> str:
        logger.info("Start transcription")

        segments, info = self.model.transcribe(
            filename,

            # -----------------------------------------------------
            # Language
            #
            # Mandarin is the primary language.
            # English words can still appear naturally in the
            # transcription.
            # -----------------------------------------------------
            language="zh",
            task="transcribe",

            # -----------------------------------------------------
            # Important anti-hallucination setting.
            #
            # Prevent a bad previous segment from contaminating
            # later segments.
            # -----------------------------------------------------
            condition_on_previous_text=False,

            # -----------------------------------------------------
            # Decoding
            # -----------------------------------------------------
            beam_size=3,
            temperature=0.0,

            # -----------------------------------------------------
            # Voice Activity Detection
            # -----------------------------------------------------
            vad_filter=True,
            #vad_parameters={
            #    "threshold": 0.5,
            #    "min_speech_duration_ms": 250,
            #    "min_silence_duration_ms": 500,
            #    "speech_pad_ms": 300,
            #},

            # -----------------------------------------------------
            # Silence / hallucination protection
            # -----------------------------------------------------
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,

            # -----------------------------------------------------
            # Mandarin + English prompt
            #
            # This is only a hint, not a hard vocabulary filter.
            # -----------------------------------------------------
            #initial_prompt=(
            #    "这是一段普通话和英语混合的语音。"
            #    "请准确转写说话内容。"
            #    "中文保持中文，英文保持英文。"
            #    "保留英文单词、英文缩写、"
            #    "软件名称、产品名称和技术术语。"
            #),
        )

        logger.info(
            f"Detected language: {info.language}, "
            f"probability: {info.language_probability:.2f}"
        )

        message_parts = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            # -----------------------------------------------------
            # Do NOT log segment.text.
            #
            # We only log numerical diagnostic information.
            # -----------------------------------------------------
            logger.debug(
                f"Segment "
                f"{segment.start:.2f}s -> {segment.end:.2f}s, "
                f"prob={segment.avg_logprob:.3f}, "
                f"no_speech={segment.no_speech_prob:.3f}, "
                f"compression={segment.compression_ratio:.3f}"
            )

            message_parts.append(text)

        return "".join(message_parts).strip()


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
async def main():
    need_import_keys = False

    # -------------------------------------------------------------
    # config.json
    # -------------------------------------------------------------
    if os.path.exists("config.json"):

        with open(
            "config.json",
            "r",
            encoding="utf-8",
        ) as fp:
            config = json.load(fp)

        bot = Bot(
            homeserver=config.get("homeserver"),
            user_id=config.get("user_id"),
            password=config.get("password"),
            device_id=config.get("device_id"),
            room_id=config.get("room_id"),
            access_token=config.get("access_token"),
            import_keys_path=config.get("import_keys_path"),
            import_keys_password=config.get("import_keys_password"),
            model_size=config.get("model_size"),
            device=config.get("device"),
            compute_type=config.get("compute_type"),
            cpu_threads=config.get("cpu_threads"),
            num_workers=config.get("num_workers"),
            download_root=config.get("download_root"),
        )

        if (
            config.get("import_keys_path")
            and config.get("import_keys_password") is not None
        ):
            need_import_keys = True

    # -------------------------------------------------------------
    # Environment variables
    # -------------------------------------------------------------
    else:

        bot = Bot(
            homeserver=os.environ.get("HOMESERVER"),
            user_id=os.environ.get("USER_ID"),
            password=os.environ.get("PASSWORD"),
            device_id=os.environ.get("DEVICE_ID"),
            room_id=os.environ.get("ROOM_ID"),
            access_token=os.environ.get("ACCESS_TOKEN"),
            import_keys_path=os.environ.get("IMPORT_KEYS_PATH"),
            import_keys_password=os.environ.get(
                "IMPORT_KEYS_PASSWORD"
            ),
            model_size=os.environ.get("MODEL_SIZE"),
            device=os.environ.get("DEVICE"),
            compute_type=os.environ.get("COMPUTE_TYPE"),
            cpu_threads=int(
                os.environ.get("CPU_THREADS", 0)
            ),
            num_workers=int(
                os.environ.get("NUM_WORKERS", 1)
            ),
            download_root=os.environ.get("DOWNLOAD_ROOT"),
        )

        if (
            os.environ.get("IMPORT_KEYS_PATH")
            and os.environ.get("IMPORT_KEYS_PASSWORD") is not None
        ):
            need_import_keys = True

    # -------------------------------------------------------------
    # Login
    # -------------------------------------------------------------
    await bot.login()

    # -------------------------------------------------------------
    # Import encryption keys if configured
    # -------------------------------------------------------------
    if need_import_keys:
        logger.info(
            "Starting encryption key import"
        )
        await bot.import_keys()

    # -------------------------------------------------------------
    # Matrix sync
    # -------------------------------------------------------------
    sync_task = asyncio.create_task(
        bot.sync_forever()
    )

    # -------------------------------------------------------------
    # Signal handling
    # -------------------------------------------------------------
    loop = asyncio.get_running_loop()

    for signame in (
        "SIGINT",
        "SIGTERM",
    ):
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda: asyncio.create_task(
                bot.close(sync_task)
            ),
        )

    # -------------------------------------------------------------
    # Upload Matrix encryption keys if required
    # -------------------------------------------------------------
    if bot.client.should_upload_keys:
        await bot.client.keys_upload()

    await sync_task


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Bot started!")
    asyncio.run(main())
