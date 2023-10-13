/*
 * Agent runs inside the container to execute user function code. Once a response
 * is sent back to client. Container should be frozen, which means all activities,
 * including agent itself, will be frozen as well. Container will be defrosted by
 * client (EA) when the next invocaiton is ready.
 *
 * Status code
 *   - 200: Agent will always return 200 once function is executed. Execution
 *          result will be saved in the response body.
 *   - 400: Indicate the request body is invalid.
 *   - 404: Function handler is invalid.
 *   - 417: Handled function error.
 *   - 429: Too many requests.
 *   - 500: Agent is unable to process the request.
 *
 * Environment variables
 *   - FC_SERVER_PORT: Server port number.
 *   - FC_FUNC_CODE_PATH: The location of the function code.
 */
'use strict';
var express = require('express');
var bodyParser = require('body-parser');
// Utils
var configx = require('./config.js');
var logger = require('./logger.js');
var conxole = require('./console.js');
var eventLoop = require('./event_loop.js');
var sanitizer = require('./sanitizer.js');
var npp = require('./npp.js');
// Handlers
var setup = require('./setup.js');
var responseParser = require('./response_parser.js');
var validator = require('./validator.js');
var throttle = require('./throttle.js');
var caRequest = require('./invoke.js');
var prepare_code = require('./prepare_code.js')

// Configuration.
const config = configx.load();

// Set up logger.
logger.config(config);

// Redirect console log to a log file.
conxole.config(config);
conxole.redirect();

// Enable sanitizer to wash sensitive information.
sanitizer.config(config);
sanitizer.washErrorStack();
sanitizer.washEnv();

// start agenthub for alinode
if (process.alinode) {
  npp.startAgenthub(process.env.APP_ID, process.env.APP_SECRET);
}

if (process.env.ENABLE_REMOTE_DEBUG === 'true') {
    require('./pandora/pandora').start()
        .then(() => {
            console.log('Pandora.js remote debug client started, waiting for open inspector.');
        })
        .catch((err) => {
            console.error('Pandora.js remote debug client start failed, ', err);
        });
}

// Set up server.
var app = express();
app.post('/invoke', [setup(config), responseParser, validator.validateReqHeader, throttle, caRequest]);
app.post('/initialize', [setup(config), responseParser, validator.validateReqHeader, throttle, caRequest]);
app.get('/pre-stop', [setup(config), responseParser, validator.validateReqHeader, throttle, caRequest]);
app.get('/pre-freeze', [setup(config), responseParser, validator.validateReqHeader, throttle, caRequest]);
// add a new api to load code from disk into memory
app.post('/prepare_code', [setup(config), responseParser, validator.validatePrepareCode, throttle, prepare_code.prepareFunction]);

// Validate configuration.
if (!config.server.port || config.server.port < 0 || config.server.port > 65536) {
    logger.getLogger().error('Port number must range from 0 to 65536: %s.', config.server.port);
    return;
}
if (!config.func.codePath) {
    logger.getLogger().error('Function code path is not set.');
    return;
}

// Start server.
var server = app.listen(config.server.port, function() {
    logger.getLogger().info('FunctionCompute nodejs runtime inited.');
    logger.getLogger().info('Function code path is set on: %s.', config.func.codePath);
    require('dns').lookup(require('os').hostname(), function (err, add, fam) {
          logger.getLogger().info('Started server, listening on %s:%s.', add, config.server.port);
    })
});
server.timeout = 0; // never timeout
// See https://nodejs.org/dist/latest-v8.x/docs/api/http.html#http_server_keepalivetimeout
server.keepAliveTimeout = 0; // keepalive, never timeout

// Catch user function error and output to function logs.
process.on('uncaughtException', function(err) {
    console.error(err.stack);
    logger.getLogger().error(err.stack);
});
