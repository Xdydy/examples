/*
 * Validator valides the input request.
 */
'use strict';
var util = require('util');
var constant = require('./constant.js');
var url = require("url");

// Headers defines a list of headers that need to be validated.
var initializeHeaders = [
    constant.HEADERS.FUNCTION_INITIALIZER,
    constant.HEADERS.FUNCTION_INITIALIZATION_TIMEOUT,
    constant.HEADERS.FUNCTION_NAME
];

var invokeHeaders = [
    constant.HEADERS.CONTENT_TYPE,
    constant.HEADERS.REQUEST_ID,
    constant.HEADERS.FUNCTION_NAME,
    constant.HEADERS.FUNCTION_HANDLER,
    constant.HEADERS.FUNCTION_MEMORY,
    constant.HEADERS.FUNCTION_TIMEOUT
];

var preStopHeaders = [
    constant.HEADERS.FUNCTION_PRE_STOP_HANDLER,
    constant.HEADERS.FUNCTION_NAME
];

var preFreezeHeaders = [
    constant.HEADERS.FUNCTION_PRE_FREEZE_HANDLER,
    constant.HEADERS.FUNCTION_NAME
];

// Headers defines a list of headers that need to be validated.
var prepareCodeHeaders = [
    constant.HEADERS.REQUEST_ID,
    constant.HEADERS.FUNCTION_HANDLER,
];

exports.validateReqHeader = function(request, response, next) {
    try {
        var pathName = url.parse(request.url).pathname;
        if (pathName === constant.INITIALIZE_PATH_NAME) {
            validateHeaders(request, initializeHeaders);
        } else if (pathName === constant.INVOKE_PATH_NAME){
            validateHeaders(request, invokeHeaders);
        } else if (pathName === constant.PRE_STOP_PATH_NAME) {
            validateHeaders(request, preStopHeaders);
        } else {
            validateHeaders(request, preFreezeHeaders);
        }

        // Check content type.
        var contentType = request.get(constant.HEADERS.CONTENT_TYPE);
        if (contentType !== 'application/octet-stream') {
            var err = new Error(util.format('Unsupported content type: %s', contentType));
            err.shouldSanitize = true;
            throw err;
        }

        // Execute next handler.
        next();
    } catch (err) {
        // Reject request.
        response._fc_send(400, err.toString());
        return
    }
    
}

exports.validatePrepareCode = function(request, response, next) {
    try {
        validateHeaders(request, prepareCodeHeaders)

        // Execute next handler.
        next();
    } catch (err) {
        // Reject request.
        response._fc_send(400, err.toString());
        return
    }

}

var validateHeaders = function(request, headers) {
    // Validate headers.
    for (var i in headers) {
        validateHeader(request, headers[i]);
    }
};

var validateHeader = function(request, headerName, headerValue) {
    var err;
    var val = request.get(headerName);
    if (!val) {
        err = new Error(util.format('Missing header %s', headerName));
        err.shouldSanitize = true;
        throw err;
    }
    if (val === '') {
        err = new Error(util.format('Invalid %s header value %s', headerName, val));
        err.shouldSanitize = true;
        throw err;
    }
};
