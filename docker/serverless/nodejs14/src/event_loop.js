'use strict';

// The period that will inspect the event loop.
var INSPECT_PERIOD = 1; // Unit is millisecond.$

/*
 * Tag an object with a fc id. If tagged object is an event
 * handle, it will be ignored/skipped while inspecting event loop.
 */
var ignore = function(obj, tag) {
    if (obj) {
        obj._fc_id = '_fc_' + (tag ? tag : '');
    }
};
exports.ignore = ignore;

/*
 * Remove fc id from object.
 */
exports.removeTag = function(obj) {
    if (obj) {
        delete obj._fc_id;
    }
};

/*
 * Return true if event loop is empty.
 */
var isEmpty = function() {
    var handles = process._getActiveHandles();
    for (var index in handles) {
        var handle = handles[index];
        // This handle is not owned by us.
        if (!handle._fc_id) {
            return false;
        }
    }
    return true;
};
exports.isEmpty = isEmpty;

// Print out handle that is not labeld with fc id in event loop.
var print = function() {
    var handles = process._getActiveHandles();
    for (var index in handles) {
        var handle = handles[index];
        if (!handle._fc_id) {
            console.log(handle);
        }
    }
};
exports.print = print;

/*
 * Invoke callback when the event loop is empty.
 */
exports.waitForEmpty = function(callback) {
    var inspector = setInterval(function() {
        if (isEmpty()) {
            clearInterval(inspector);
            callback();
        }
    }, INSPECT_PERIOD);
    inspector.unref();
    return inspector;
};
