'use strict';
var constant = require('./constant.js');
var eventLoop = require('./event_loop.js');
var httpparam = require('./httpparam.js');

// Callback object provides a callback method for returning data back to client.
// Function will be terminated once the callback is invoked.
exports.create = function(request, response, isHttpMode) {
    var isConsumed = false;
    var consumedErr = null;
    var consumedData = null;
    var isDigested = false;

    if (isHttpMode) {
        return function(resp) {
            if (!(resp instanceof httpparam.Response)) {
                // let it crash in case of invalide resp instance without error.
                throw new Error('response instance required');
            }
            if (isConsumed) {
                return;
            }
            isConsumed = true;
            process.nextTick(function() {
                if (isDigested) {
                    return;
                }
                isDigested = true;
                // handle consumeResp
                response._fc_httpContextDone(resp);
            });
        }
    }

    // Digest the consumed error and data. Digested error or data will send
    // back to the client along the response. Only the first call to digest
    // will succeed.
    var digest = function() {
        if (isDigested) {
            return;
        }
        isDigested = true;
        response._fc_contextDone(consumedErr, consumedData);
    };

    // Consume the error, data. Only the first call to consume will succeed.
    var consume = function(err, data) {
        if (isConsumed) {
            return;
        }
        isConsumed = true;
        consumedErr = err;
        consumedData = data;
        process.nextTick(digest);
    };


    // Warp the consume inside a function so that its implementation is
    // not visible to user function.
    return function(err, data) {
        consume(err, data);
    };
};
