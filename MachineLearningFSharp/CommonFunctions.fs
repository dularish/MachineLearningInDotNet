module CommonFunctions

open System
open MathNet.Numerics.LinearAlgebra

exception IncompatibleOutputMatrixDimensions of string


let sigmoid = fun(x:double) ->
    (1./(1. + (Math.E)**(-1.*x)))

let sigmoid' = fun(x:double) ->
    let sigmoidValue = sigmoid x
    sigmoidValue * (1. - sigmoidValue)
    
let relu = fun(x:double) ->
    if x > 0. then x else 0.

let relu' = fun(x:double) ->
    if x > 0. then x else 0.

let computeCost predictedValues actualValues= 
    //Not using collapseSamples helper method here as this is more efficient
    let costForAllSamplesMatrix = actualValues .* Matrix.Log(predictedValues) + (1. - actualValues) .* Matrix.Log(1.-predictedValues)

    let rowSum = costForAllSamplesMatrix.RowSums()

    if rowSum.Count = 1 then
        (-1./((double)predictedValues.ColumnCount)) * rowSum.[0]
    else
        raise (IncompatibleOutputMatrixDimensions ("Wrong input matrix"))