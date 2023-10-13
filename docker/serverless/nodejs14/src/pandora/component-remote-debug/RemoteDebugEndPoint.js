"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RemoteDebugEndPoint = void 0;
const assert = require("assert");
const util_1 = require("./util");
class RemoteDebugEndPoint {
  constructor(ctx) {
    this.prefix = '/remote-debug';
    this.ctx = ctx;
  }
  route(router) {
    const hubFacade = this.ctx.hubFacade;
    const configManager = hubFacade.getConfigManager();
    router.get('/open-port', async (ctx) => {
      try {
        const pid = ctx.query.pid;
        assert(pid, 'pid is required');
        assert(configManager, 'configManager have\'t been initialized');
        configManager.publish(util_1.genProcessSubscribeTopic(pid), true);
        ctx.ok(true);
      }
      catch (err) {
        ctx.fail(err.message);
      }
    });
    router.get('/close-port', async (ctx) => {
      try {
        const pid = ctx.query.pid;
        assert(pid, 'pid is required');
        assert(configManager, 'configManager have\'t been initialized');
        configManager.publish(util_1.genProcessSubscribeTopic(pid), false);
        ctx.ok(true);
      }
      catch (err) {
        ctx.fail(err.message);
      }
    });
  }
}
exports.RemoteDebugEndPoint = RemoteDebugEndPoint;
//# sourceMappingURL=RemoteDebugEndPoint.js.map