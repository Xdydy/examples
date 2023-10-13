'use strict';
var fs = require('fs');
var util = require('util');
var path = require('path');
var bufferBuilder = require('./buffer_builder.js');
var getRawBody = require('raw-body');
var constant = require('./constant.js');
var conxole = require('./console.js');
var context = require('./context.js');
var callback = require('./callback.js');
var httpparam = require('./httpparam.js');

var logger = require('./logger.js');
var encoding = 'utf8';

var fcCodePath = path.parse(__dirname).dir;
// Handle invoke request.
exports.prepareFunction = function(request, response) {
    // Store request id in process. This id will be added to log entry.
    // TODO: Support multiple rid in reserve mode.
    process._fc = {
        requestId: request.get(constant.HEADERS.REQUEST_ID)
    };

    process.stdout.write(constant.LOG_TAIL_START_PREFIX_PREPARE + request.get(constant.HEADERS.REQUEST_ID) + '\n');
    logger.getLogger().info(constant.LOG_TAIL_START_PREFIX_PREPARE + request.get(constant.HEADERS.REQUEST_ID));

    // Setup function handler.
    var func = null;
    try {
        var codePath = request.getConfig().func.codePath;
        var handler = request.get(constant.HEADERS.FUNCTION_HANDLER);
        func = exports.loadFunction (codePath, handler);
    } catch (err) {
        var stackTrace;
        // when loadFunction error, and error stack does not caintains user code path,
        // meanwhile, shouldSanitize will not be assigned, then this following logic is needed.
        if (err.name === 'SyntaxError') {
            stackTrace = [];
            var st = err.stack.split('\n');
            for (let i = 0; i < st.length; i++) {
                if (!st[i].includes(fcCodePath)) {
                    stackTrace.push(st[i].trim());
                }
            }
        } else {
            stackTrace =  err.stack.split('\n').map(function(line) {
                return line.trim();
            });
        }

        var msg = {
            errorMessage: err.message,
            errorType: err.name || 'Error',
            stackTrace: stackTrace
        };
        conxole.errorCap(JSON.stringify(msg));
        response._fc_send(404, JSON.stringify(msg,null,2), err);
        return null;
    }

    var isHttpMode = false;
    var cb = callback.create(request, response, isHttpMode);
    var ctx = context.create(request, response);

    var replyPrepareCode = function(handler, cb) {
        var json = {
            'code_prepared': true,
            'handler': handler,
        }
        var reply = bufferBuilder.from(JSON.stringify(json), encoding);
        cb(null, reply);
    }

    getRawBody(request, {limit:'12mb'}, function(err, body){
        // Setup event, context and callback.
        // Invoke function. We don't catch any error threw by the user
        // function. We will just let it crashes the process. EA should
        // construct an proper error message for the crash.
        if (err) {
            cb(err);
            return null;
        }
        replyPrepareCode(handler, cb)
    });

    return null;
};

var globalModule = {}; // use this map to cache loaded functions

/*
 * Reload the function definition into memory if handler is changed.
 * handle is in the format of '<file name>.<method name>'.
 * Throw: throw error if handler is invalid.
 *   - FailNotFoundError
 *   - Error
 */
exports.loadFunction = function(codePath, handler) {
    // if this handler is already loaded, just return it
    if ( globalModule[handler] != null && typeof(globalModule[handler]) === 'function') {
        return globalModule[handler]
    }

    process.stdout.write("load code for handler:" + handler + '\n')
    logger.getLogger().info("load code for handler:" + handler + '\n')

    var err;
    var index = handler.lastIndexOf('.');
    if (index === -1) {
        err = new Error(util.format('Invalid handler \'%s\'', handler));
        err.shouldSanitize = true;
        throw err;
    }
    var moduleName = handler.slice(0, index);
    var handlerName = handler.slice(index + 1);
    var modulePath = path.join(codePath, moduleName + '.js');
    if (!fs.existsSync(modulePath)) {
      err = new Error(util.format('Module \'%s\' is missing.', modulePath));
      err.shouldSanitize = true;
      throw err;
    }
    var module = require(modulePath);

    if (typeof(module[handlerName]) === 'function') {
        globalModule[handler] = module[handlerName] // cache loaded function to globalModule
        return module[handlerName];
    }
    err = new Error(util.format('Handler \'%s\' is missing on module \'%s\'', handlerName, moduleName));
    err.shouldSanitize = true;
    throw err;
};
