/*
 * Context is the object that user function can access to get function
 * metadata.
 */
'use strict';
var constant = require('./constant.js');
var winston = require('winston');
var util = require('util');

function ContextLogger (reqId) {
    var that = this;
    that.transport = new (winston.transports.Console)({
        level: 'silly',
        json: false,
        timestamp: function () {
            return new Date().toISOString();
        },
        formatter: function (options) {
            return util.format('%s %s [%s] %s',
                options.timestamp(),
                reqId,
                options.level,
                options.message ? options.message : '');
        }
    });
    var logger = new (winston.Logger)({
        transports: [that.transport],
    });
    logger.setLogLevel = function (level) {
        that.transport.level = level;
    }

    return logger;
}

//Context object provide function medata information.
exports.create = function (request, response) {
    var reqId = request.get(constant.HEADERS.REQUEST_ID);

    var ctx = {
        'requestId': reqId,
        'credentials': {
            accessKeyId: request.get(constant.HEADERS.ACCESS_KEY_ID),
            accessKeySecret: request.get(constant.HEADERS.ACCESS_KEY_SECRET),
            securityToken: request.get(constant.HEADERS.SECURITY_TOKEN)
        },
        'function': {
            name: request.get(constant.HEADERS.FUNCTION_NAME),
            handler: request.get(constant.HEADERS.FUNCTION_HANDLER),
            memory: parseInt(request.get(constant.HEADERS.FUNCTION_MEMORY)),
            timeout: parseInt(request.get(constant.HEADERS.FUNCTION_TIMEOUT)),
            initializer: request.get(constant.HEADERS.FUNCTION_INITIALIZER),
            initializationTimeout: parseInt(request.get(constant.HEADERS.FUNCTION_INITIALIZATION_TIMEOUT))
        },
        'service': {
            name: request.get(constant.HEADERS.SERVICE_NAME),
            logProject: request.get(constant.HEADERS.SERVICE_LOG_PROJECT),
            logStore: request.get(constant.HEADERS.SERVICE_LOG_STORE),
            qualifier: request.get(constant.HEADERS.QUALIFIER),
            versionId: request.get(constant.HEADERS.VERSION_ID)
        },
        'region': request.get(constant.HEADERS.REGION),
        'accountId': request.get(constant.HEADERS.ACCOUNT_ID),
        'logger': new ContextLogger(reqId),
        'retryCount': 0,
        'tracing': {
            'openTracingSpanContext': request.get(constant.HEADERS.OPENTRACING_SPAN_CONTEXT),
            'openTracingSpanBaggages': parseOpenTracingBaggages(request),
            'jaegerEndpoint': request.get(constant.HEADERS.JAEGER_ENDPOINT)
        }
    };

    if (request.get(constant.HEADERS.RETRY_COUNT)) {
        ctx['retryCount'] = parseInt(request.get(constant.HEADERS.RETRY_COUNT))
    }

    return ctx;
};

function parseOpenTracingBaggages (request) {
    var base64Baggages = request.get(constant.HEADERS.OPENTRACING_SPAN_BAGGAGES);
    var baggages = {}
    if (base64Baggages != undefined && base64Baggages != '') {
        try {
            // import when used
            var bufferBuilder = require('./buffer_builder.js');
            var logger = require('./logger.js');
            baggages = JSON.parse(bufferBuilder.from(base64Baggages, 'base64').toString());
        } catch (e) {
            logger.getLogger().error('Failed to parse base64 opentracing baggages', e);
        }
    }
    return baggages
}