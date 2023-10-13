/*
 * Throttle the incoming request.
 */
'use strict';
var limit = 100;

module.exports = exports = function(request, response, next) {
    var cache = request.getConfig().cache;

    if (cache.concurrent == null) {
        cache.concurrent = 0;
    }

    if (cache.concurrent < limit) {
        // Accept request.
        cache.concurrent++;
        var send = response.send.bind(response);
        response.send = function(body) {
            send(body);
            cache.concurrent--;
        };

        // Execute next handler.
        next();
    } else {
        // Reject request.
        response._fc_send(429, 'Too many requests.');
        return;

    }
};
