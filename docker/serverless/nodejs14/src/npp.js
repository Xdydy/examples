
/*
 * npp starts the agenthub for alinode
 * appid and secret is provided by env APP_ID and APP_SECRET when start cagent
 */

'use strict';
var fs = require('fs');
var spawn = require('child_process').spawn;
var out = fs.openSync('/tmp/agenthubout.log', 'a');

exports.startAgenthub = function(appid, secret) {
  if (!appid || !secret) {
    return;
  }

  var configPath = '/tmp/agenthub.config.json';
  fs.writeFileSync(configPath, JSON.stringify({appid: appid, secret: secret}));
  var env = Object.create(process.env);
  env.ENABLE_NODE_LOG = 'NO';
  env.HOME = '/tmp'

  spawn('/usr/local/bin/agenthub', ['start', configPath], {stdio: [out, out, out],  env: env, detached: true});
};
