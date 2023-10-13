'use strict';

/*
 * if Node.js support Buffer.from then use Buffer.from else use new Buffer
 * https://nodejs.org/fr/docs/guides/buffer-constructor-deprecation/#variant-3
 */
exports.from = function(notNumber, encoding) {
    if (Buffer.from && Buffer.from !== Uint8Array.from) {
        return Buffer.from(notNumber, encoding);
    } else {
        if (typeof notNumber === 'number') {
            throw new Error('The "size" argument must be not of type number.');
        }
        return new Buffer(notNumber, encoding);
    }
};

/*
 * if Node.js support Buffer.alloc then use Buffer.alloc else use new Buffer
 */
exports.alloc = function(size) {
    const buf = Buffer.alloc ? Buffer.alloc(size) : new Buffer(size).fill(0);
    return buf
};
