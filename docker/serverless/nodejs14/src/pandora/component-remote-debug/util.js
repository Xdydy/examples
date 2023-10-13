"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.genProcessSubscribeTopic = void 0;
function genProcessSubscribeTopic(pid) {
  pid = pid || process.pid.toString();
  return 'open_debugger_' + pid;
}
exports.genProcessSubscribeTopic = genProcessSubscribeTopic;
//# sourceMappingURL=util.js.map