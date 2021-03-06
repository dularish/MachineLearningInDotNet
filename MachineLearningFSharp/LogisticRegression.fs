﻿module LogisticRegression

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics
open MathNet.Numerics
open System
open MathNet.Numerics.Data.Text
open MathNet.Numerics.LinearAlgebra

exception IncompatibleOutputMatrixDimensions of string
exception IncompatibleTypeForWeights of string
exception IncompatibleTypeForBias of string
exception IncompatibleTypeOutputForWeights of string
exception IncompatibleTypeOutputForBias of string
exception UnexpectedDimensionDuringCalculation of string

let initializeParameters n = 
    let W = Matrix<double>.Build.Dense(n,1,0.)
    let b = 0.0
    (W,b)

let computeProbabilityOfSuccess (W:Matrix<double>) (b:double) (X:Matrix<double>) =
    let computedMatrix = 
        (W.Transpose() * X ) + b
        |> Matrix.map (fun(a) -> CommonFunctions.sigmoid(a))
    if computedMatrix.RowCount = 1 then
        computedMatrix
    else
        raise (UnexpectedDimensionDuringCalculation "Output dimension is not single row")

let computeDerivatives (Y':Matrix<double>) (Y:Matrix<double>) X = 
    let dw = (X * (Y' - Y).Transpose())
                |> Matrix.map (fun a -> a * (1./(double)Y.ColumnCount))
    let db = (1./(double)Y.ColumnCount) * ((Y' - Y).RowSums()).[0]

    dw, db

let initialize_dataset = 
    MLDatasets.load_catsDatasets()

type DerivativeTypes =
    |MatrixDerivative of Matrix<float>
    |FloatDerivative of float

let propagate W b X Y =
    let Y' = computeProbabilityOfSuccess W b X
    let cost = CommonFunctions.computeCost Y' Y
    let dw,db = computeDerivatives Y' Y X
    let grads = dict<string, DerivativeTypes>["dw",MatrixDerivative dw; "db", FloatDerivative db]
    grads, cost


let predict_class W b X = 
    let Y' = 
        (computeProbabilityOfSuccess W b X).Row(0)
        |> Vector.map (fun x -> if x <= 0.5 then 0. else 1.)
    Y'


let optimize W b X Y num_iterations learning_rate =
    let mutable Wm = W
    let mutable bm = b
    let mutable costs = []
    [0..num_iterations]
    |> List.iter (fun(i) ->
                    let grads, cost = propagate Wm bm X Y
                    costs <- costs @ [cost]
                    let dw = 
                        match grads.["dw"] with
                        | MatrixDerivative someMatrix ->
                            someMatrix
                        | _ -> raise (IncompatibleTypeForWeights ("Unexpected"))
                    let db =
                        match grads.["db"] with
                        | FloatDerivative someFloat ->
                            someFloat
                        | _ -> raise (IncompatibleTypeForWeights ("Unexpected"))

                    Wm <- Wm - (learning_rate * dw)
                    bm <- bm - (learning_rate * db)

                    if i%100 = 0 then
                        printfn "Cost after %A iterations : %A" i cost
                    )

    let trained_params = dict<string, DerivativeTypes>["W",MatrixDerivative Wm;"b",FloatDerivative bm]

    trained_params, costs

let createModel (X_train:Matrix<double>) Y_train X_test (Y_test:Matrix<float>) (num_iterations) learning_rate =
    let W,b = initializeParameters X_train.RowCount

    let trained_params,costs = optimize W b X_train Y_train num_iterations learning_rate

    let trained_W = 
        match trained_params.["W"] with
        | MatrixDerivative matrixValue ->
            matrixValue
        | _ ->
            raise (IncompatibleOutputMatrixDimensions "Optimize function has returned non-matrix type of W")
    let trained_b = 
        match trained_params.["b"] with
        | FloatDerivative floatValue ->
            floatValue
        | _ ->
            raise (IncompatibleTypeOutputForBias "Optimize function has returned non-float type of b")

    let pred_train = predict_class trained_W trained_b X_train
    let pred_test = predict_class trained_W trained_b X_test

    let accuracy_train = CommonFunctions.computeAccuracy pred_train (Y_train.Row(0))
    let accuracy_test = CommonFunctions.computeAccuracy pred_test (Y_test.Row(0))

    printfn "Accuracy of training samples : %A" accuracy_train
    printfn "Accuracy of testing samples : %A" accuracy_test

    trained_params
