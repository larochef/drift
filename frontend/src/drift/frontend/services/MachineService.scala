package drift.frontend.services

import drift.shared.MachineStatus

import com.raquo.laminar.api.L.*

/** The machine's load and memory as the status socket last said them, for the
  * sidebar and under every running bar. Nothing to load and nothing to mount:
  * the shell holds the socket, and the sidebar keeps this signal read on every
  * page.
  */
class MachineService(statusSocket: StatusSocketService) {

  /** `None` until the socket has spoken, and where the backend cannot read the
    * machine.
    */
  val status: Signal[Option[MachineStatus]] =
    statusSocket.machine.map(Some(_)).startWith(None)
}
