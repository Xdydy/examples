/*
 * Inject meta into request/response object.
 */
'use strict';
var logger = require('./logger.js');
var eventLoop = require('./event_loop.js');

module.exports = exports = function(config) {
    return function(request, response, next) {
        // Allow other handlers to get the config.
        request.getConfig = function() {
            return config;
        };

        /*
         * Set the fc id for our emitter object. Object will fc id will be
         * ignored when we inspect the event loop.
         */
        eventLoop.ignore(request.socket._server, 'server');
        eventLoop.ignore(request.socket, 'socket');
        eventLoop.ignore(process.stdin, 'stdin');
        eventLoop.ignore(process.stdout, 'stdout');
        eventLoop.ignore(process.stderr, 'stderr');

        // Execute next handler.
        next();
    };
};
