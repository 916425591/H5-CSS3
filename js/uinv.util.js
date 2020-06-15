var uinv = {};

/**
 * 合并对象 只会在A的基础上添加元素,不影响原有元素 返回长大了的A
 * @param {Object} opObjectA
 * @param {Object} opObjectB
 * @param {Boolean} isDeep
 * @param {Boolean} isReturnNew
 * @param {Boolean} isCloneObjDeep
 * @return {Object}
 */
uinv.combineNew = (function () {
    var fun = function (opObjectA, opObjectB, isDeep, isReturnNew, isCloneObjDeep) {
        for (var cur in opObjectB) {
            if (isDeep) {
                if (opObjectA[cur] !== undefined && opObjectA[cur] !== null
                    && !(opObjectA[cur] instanceof Array) && typeof opObjectA[cur] == 'object'
                    && !(opObjectB[cur] instanceof Array) && typeof opObjectB[cur] == 'object') {
                    fun(opObjectA[cur], opObjectB[cur], isDeep, false);
                } else {
                    if (opObjectA[cur] === undefined || opObjectA[cur] === null) opObjectA[cur] = opObjectB[cur];
                }
            } else {
                if (opObjectA[cur] === undefined || opObjectA[cur] === null) opObjectA[cur] = opObjectB[cur];
            }
        }
        return opObjectA;
    };
    return fun;
})();

/**
 * 合并对象 A中与B相同名称的元素会被替换成B中的值 返回长大了的A
 * @param {Object} opObjectA
 * @param {Object} opObjectB
 * @param {Boolean} isDeep
 * @param {Boolean} isReturnNew
 * @param {Boolean} isCloneObjDeep
 * @return {Object}
 */
uinv.combine = (function () {
    var fun = function (opObjectA, opObjectB, isDeep, isReturnNew, isCloneObjDeep) {
        if (isReturnNew) {
            var tempFun = uinv.util.cloneObj || uinv.cloneObj;
            var result = tempFun(opObjectA, isCloneObjDeep);
            fun(result, opObjectB, isDeep, false);
            return result;
        }
        for (var cur in opObjectB) {
            if (isDeep) {
                if (opObjectA[cur] !== undefined && opObjectA[cur] !== null
                    && !(opObjectA[cur] instanceof Array) && typeof opObjectA[cur] == 'object'
                    && !(opObjectB[cur] instanceof Array) && typeof opObjectB[cur] == 'object') {
                    console.log(1)
                    fun(opObjectA[cur], opObjectB[cur], isDeep, false);
                } else {
                    console.log(2)
                    opObjectA[cur] = opObjectB[cur];
                }
            } else {
                opObjectA[cur] = opObjectB[cur];
            }
        }
        return opObjectA;
    };
    return fun;
})();

/**
 * 获取第一个元素的名称
 * @param {Object} opObject
 * @return {String}
 */
uinv.getFirstKey = function (opObject) {
    for (var i in opObject) {
        return i;
    }
    return null;
};

/**
 * 平铺数组中数组
 * @param {Array} arr 数组
 * @return {Array}
 */
uinv.flatten = function flatten(arr) {
    return arr.reduce(function (a, b) {
        console.log(a,'111111',b)
        return a.concat(Array.isArray(b) ? flatten(b) : b);
    }, []);
};
