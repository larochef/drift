package drift.backend.session

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker

/** Launch tunables, read fresh on every launch so a hand-edit of
  * `~/.config/drift/settings/sessions.json` applies without a restart. There is
  * no UI for these yet; the defaults are the design.
  */
case class SessionSettings(
    /** These models fill VRAM; a second concurrent load will usually OOM, so
      * one at a time by default (`specs/07-launch-and-supervision.md`).
      */
    maximumConcurrentSessions: Int = 1,
    /** Assistant (llama.cpp) sessions are counted apart: a chat model beside a
      * loaded sd-server is the normal case on unified memory
      * (`specs/18-assistant-models-and-sessions.md`).
      */
    maximumConcurrentAssistantSessions: Int = 1,
    portRangeStart: Int = 7860,
    portRangeEnd: Int = 7899,
    /** How long `starting` may last before the launch is failed — generous,
      * because a load can spend minutes on 30GB of weights
      * (`specs/13-log-streaming.md`).
      */
    readinessTimeoutMinutes: Int = 15
)
object SessionSettings {
  given JsonValueCodec[SessionSettings] = JsonCodecMaker.make
}
