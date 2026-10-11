package drift.backend.session

import drift.shared.storedRecord

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
    maximumConcurrentSessions: Int,
    /** Assistant (llama.cpp) sessions are counted apart: a chat model beside a
      * loaded sd-server is the normal case on unified memory
      * (`specs/18-assistant-models-and-sessions.md`).
      */
    maximumConcurrentAssistantSessions: Int,
    portRangeStart: Int,
    portRangeEnd: Int,
    /** How long `starting` may last before the launch is failed — generous,
      * because a load can spend minutes on 30GB of weights
      * (`specs/13-log-streaming.md`).
      */
    readinessTimeoutMinutes: Int,
    /** Under this share of the memory left once a model is loaded, its session
      * warns that generations may crawl (`MemoryHeadroom`).
      */
    memoryHeadroomWarningPercent: Int
)
object SessionSettings {

  /** What a machine starts with, until `settings/sessions.json` says otherwise.
    */
  val initial: SessionSettings = SessionSettings(
    maximumConcurrentSessions = 1,
    maximumConcurrentAssistantSessions = 1,
    portRangeStart = 7860,
    portRangeEnd = 7899,
    readinessTimeoutMinutes = 15,
    memoryHeadroomWarningPercent = 10
  )

  given JsonValueCodec[SessionSettings] = JsonCodecMaker.make(storedRecord)
}
