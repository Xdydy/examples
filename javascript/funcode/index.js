const add = (x) => (x+3)
const mul = (x) => (x*3)

const pipe = (...functions) => (input) => functions.reduce((acc,fun)=>fun(acc), input)

const addMul = pipe(
    add,mul
)

console.log(addMul(5));