'use strict';
var fs = require('fs');
var util = require('util');
var path = require('path');
var getRawBody = require('raw-body');
var constant = require('./constant.js');
var conxole = require('./console.js');
var context = require('./context.js');
var callback = require('./callback.js');
var httpparam = require('./httpparam.js');
var logger = require('./logger.js');
var prepare_code = require('./prepare_code.js');
var url = require("url");

var fcCodePath = path.parse(__dirname).dir;
// Handle initialize/invoke/preFreeze/preStop request
module.exports = exports = function(request, response) {
    // Store request id in process. This id will be added to log entry.
    // TODO: Support multiple rid in reserve mode.
    process._fc = {
        requestId: request.get(constant.HEADERS.REQUEST_ID)
    };

    var pathName = url.parse(request.url).pathname;
    if (pathName === constant.INITIALIZE_PATH_NAME) {
        process.stdout.write(constant.LOG_TAIL_START_PREFIX_INIITALIZE + request.get(constant.HEADERS.REQUEST_ID) + '\n');
        logger.getLogger().info(constant.LOG_TAIL_START_PREFIX_INIITALIZE + request.get(constant.HEADERS.REQUEST_ID));
    } else if (pathName === constant.INVOKE_PATH_NAME) {
        process.stdout.write(constant.LOG_TAIL_START_PREFIX_INVOKE + request.get(constant.HEADERS.REQUEST_ID) + '\n');
        logger.getLogger().info(constant.LOG_TAIL_START_PREFIX_INVOKE + request.get(constant.HEADERS.REQUEST_ID));
    } else if (pathName === constant.PRE_STOP_PATH_NAME) {
        // pathName = /preStop
        process.stdout.write(constant.LOG_TAIL_START_PREFIX_PRE_STOP + request.get(constant.HEADERS.REQUEST_ID) + '\n');
        logger.getLogger().info(constant.LOG_TAIL_START_PREFIX_PRE_STOP + request.get(constant.HEADERS.REQUEST_ID));
    } else {
        process.stdout.write(constant.LOG_TAIL_START_PREFIX_PRE_FREEZE + request.get(constant.HEADERS.REQUEST_ID) + '\n');
        logger.getLogger().info(constant.LOG_TAIL_START_PREFIX_PRE_FREEZE + request.get(constant.HEADERS.REQUEST_ID));
    }

    // Setup function handler.
    var func = null;
    try {
        var handler = null;
        var codePath = request.getConfig().func.codePath;
        if (pathName === constant.INITIALIZE_PATH_NAME) {
            handler = request.get(constant.HEADERS.FUNCTION_INITIALIZER);
        } else if (pathName === constant.INVOKE_PATH_NAME) {
            handler = request.get(constant.HEADERS.FUNCTION_HANDLER);
        } else if (pathName === constant.PRE_STOP_PATH_NAME) {
            handler = request.get(constant.HEADERS.FUNCTION_PRE_STOP_HANDLER);
        } else {
            handler = request.get(constant.HEADERS.FUNCTION_PRE_FREEZE_HANDLER);
        }
        func = prepare_code.loadFunction(codePath, handler);
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

    var httpParams = request.get(constant.HEADERS.HTTP_PARAMS);
    var isHttpMode = typeof httpParams == 'string';

    var cb = callback.create(request, response, isHttpMode);
    var ctx = context.create(request, response);

    if (isHttpMode) {
        var req = httpparam.parseRequest(request, httpParams);
        var resp = new httpparam.Response(cb);
        func(req, resp, ctx);
        return;
    }

    if ((pathName === constant.INITIALIZE_PATH_NAME) || (pathName === constant.PRE_FREEZE_PATH_NAME) ||
        (pathName === constant.PRE_STOP_PATH_NAME)){
        getRawBody(request, {limit: '0mb'}, function(err){
            if (err) {
                cb(err);
                return null;
            }
            func(ctx, cb);
        });
    } else {
        // pathName = /invoke
        getRawBody(request, {limit:'12mb'}, function(err, body){
            // Setup event, context and callback.
            // Invoke function. We don't catch any error threw by the user
            // function. We will just let it crashes the process. EA should
            // construct an proper error message for the crash.
            if (err) {
                cb(err);
                return null;
            }
            func(body, ctx, cb);
        });
    }

};
