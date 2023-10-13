/*
 * Parse response into a pretty format.
 */
'use strict';
var util = require('util');
var logger = require('./logger.js');
var bufferBuilder = require('./buffer_builder.js');
var constant = require('./constant.js');
var conxole = require('./console.js');
var httpparam = require('./httpparam.js');
var getRawBody = require('raw-body');
var stream = require('stream');
var encoding = 'utf8';

module.exports = exports = function(request, response, next) {
    // check url to see if this is a invoke or prepare_code request
    var isPrepareCode = false;
    var logPrefix = constant.LOG_TAIL_END_PREFIX_INVOKE;
    if (request.url === '/prepare_code') {
        isPrepareCode = true;
        logPrefix = constant.LOG_TAIL_END_PREFIX_PREPARE;
    } else if (request.url === '/initialize') {
        logPrefix = constant.LOG_TAIL_END_PREFIX_INITIALIZE;
    } else if (request.url === '/pre-stop') {
        logPrefix = constant.LOG_TAIL_END_PREFIX_PRE_STOP;
    } else if (request.url === '/pre-freeze') {
        logPrefix = constant.LOG_TAIL_END_PREFIX_PRE_FREEZE;
    }

    // Send back a response to client.
    response._fc_send = function(statusCode, data, err) {
        // Log
        if (statusCode !== 200) {
            logger.getLogger().warn(statusCode, data.toString());
            if (err) {
                logger.getLogger().warn(err.stack);
            }
            var msg = logPrefix + request.get(constant.HEADERS.REQUEST_ID) + ', Error: Handled function error';
            process.stdout.write(msg+"\n")
            logger.getLogger().info(msg)
        } else {
            process.stdout.write(logPrefix + request.get(constant.HEADERS.REQUEST_ID) + '\n')
            logger.getLogger().info(logPrefix + request.get(constant.HEADERS.REQUEST_ID))
        }
        // Send result back.
        response.status(statusCode);
        response.send(data);
    };

    response._fc_httpContextDone = function(resp) {
        response.status(200);
        var httpParams = httpparam.dumpHTTPParams(resp);
        response.set(constant.HEADERS.HTTP_PARAMS, httpParams);
        if (resp.body instanceof stream.Readable) {
            getRawBody(resp.body, {limit:'12mb'}, function(err, body){
                if (err) {
                    response._fc_contextDone(err)
                    return;
                }
                process.stdout.write(logPrefix + request.get(constant.HEADERS.REQUEST_ID) + '\n')
                logger.getLogger().info(logPrefix + request.get(constant.HEADERS.REQUEST_ID))
                response.send(body);
            })
            return;
        }
        process.stdout.write(logPrefix + request.get(constant.HEADERS.REQUEST_ID) + '\n')
        logger.getLogger().info(logPrefix + request.get(constant.HEADERS.REQUEST_ID))
        var output = formatData(resp.body)
        response.send(output);
    };

    // Send back a context.done() response to client.
    response._fc_contextDone = function(err, data) {
        if (err != null) {
            // Send back a response indicates there is a handled error.
            var output = formatErr(err);
            conxole.errorCap(output.toString());
            response._fc_send(417, output);
        } else {
            // Send back a response indicates invoke successfully.
            var output = formatData(data);
            response._fc_send(200, output);
        }
    };

    // Execute next handler.
    next();
};

// Translate error into a pretty object.
var formatErr = function(err) {
    var output = {};
    if (err instanceof Error) {
        output = {
            errorMessage: err.message,
            errorType: err.name,
            stackTrace: err.stack.split('\n').slice(1).map(function(line) {
                return line.trim();
            })
        };
    } else {
        output = {
            errorMessage: formatData(err).toString()
        };
    }
    return bufferBuilder.from(JSON.stringify(output), encoding);
};

// Translate data into a byte buffer.
var formatData = function(data) {
    // data is null or undefined.
    if (data == null) {
        return null;
    }

    // Buffer
    if (data instanceof Buffer) {
        return data
    }

    // Convert other data type to buffer.
    var output = data.toString();
    switch (typeof(data)) {
        case 'function':
            output = data.constructor.toString();
            break;
        case 'object':
            output = JSON.stringify(data);
            break;
        case 'string':
            output = data;
            break;
        case 'number':
        case 'boolean':
            output = data.toString();
            break;
    }
    return bufferBuilder.from(output, encoding);
};
