"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
  var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
  if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
  else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
  return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
const pandora_component_decorator_1 = require("@pandorajs/component-decorator");
const RemoteDebugEndPoint_1 = require("./RemoteDebugEndPoint");
const util_1 = require("./util");
let ComponentRemoteDebug = class ComponentRemoteDebug {
  constructor(ctx) {
    this.ctx = ctx;
    this.config = ctx.config.remoteDebug || {};
  }
  async start() {
    const { hubFacade, logger } = this.ctx;
    const configClient = hubFacade.getConfigClient();
    if (!configClient) {
      return;
    }
    configClient.subscribe(util_1.genProcessSubscribeTopic(), (value) => {
      var _a, _b, _c;
      try {
        const inspector = require('inspector');
        if (value && !inspector.url()) {
          inspector.open(8000, '0.0.0.0', false);
        }
        else if (!value && !!inspector.url()) {
          inspector.close();
        }
      }
      catch (err) {
        logger.error(err);
      }
    });
  }
  async startAtSupervisor() {
    const endPointManager = this.ctx.endPointManager;
    endPointManager.register(new RemoteDebugEndPoint_1.RemoteDebugEndPoint(this.ctx));
  }
};
ComponentRemoteDebug = __decorate([
  pandora_component_decorator_1.componentName('remoteDebug'),
  pandora_component_decorator_1.dependencies(['ipcHub', 'actuatorServer'])
], ComponentRemoteDebug);
exports.default = ComponentRemoteDebug;
//# sourceMappingURL=ComponentRemoteDebug.js.map