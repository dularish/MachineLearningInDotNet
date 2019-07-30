module LogisticRegression

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics
open MathNet.Numerics
open System
open MathNet.Numerics.Data.Text

exception IncompatibleOutputMatrixDimensions of string
exception IncompatibleTypeForWeights of string
exception IncompatibleTypeForBias of string
exception IncompatibleTypeOutputForWeights of string
exception IncompatibleTypeOutputForBias of string
exception UnexpectedDimensionDuringCalculation of string

let sigmoid = fun(x:double) ->
    (1./(1. + (Math.E)**(-1.*x)))

let initializeParameters n = 
    //let W = Matrix<double>.Build.Random(n,1,9)
    let W = Matrix<double>.Build.Dense(n,1,0.)
    let b = 0.0
    (W,b)

let computeProbabilityOfSuccess (W:Matrix<double>) (b:double) (X:Matrix<double>) =
    let computedMatrix = 
        (W.Transpose() * X ) + b
        |> Matrix.map (fun(a) -> sigmoid(a))
    if computedMatrix.RowCount = 1 then
        computedMatrix
    else
        raise (UnexpectedDimensionDuringCalculation "Output dimension is not single row")

let computeCost predictedValues actualValues= 
    let firstLogTerm = 
        predictedValues
        |> Matrix.map (fun(a) -> log a)
    let secondLogTerm = 
        predictedValues
        |> Matrix.map (fun(a) -> log (1.-a))
        
    let costForAllSamplesMatrix = actualValues .* firstLogTerm + (actualValues |> Matrix.map (fun a -> 1. - a)) .* secondLogTerm

    let rowSum = costForAllSamplesMatrix.RowSums()

    if rowSum.Count = 1 then
        (-1./((double)predictedValues.ColumnCount)) * rowSum.[0]
    else
        raise (IncompatibleOutputMatrixDimensions ("Wrong input matrix"))

let computeDerivatives (predictedValues:Matrix<double>) (actualValues:Matrix<double>) X = 
    let dw = (X * (predictedValues - actualValues).Transpose())
                |> Matrix.map (fun a -> a * (1./(double)actualValues.ColumnCount))
    let db = (1./(double)actualValues.ColumnCount) * ((predictedValues - actualValues).RowSums()).[0]

    dw, db

let initialize_dataset = 
    MLDatasets.load_catsDatasets()

type DerivativeTypes =
    |MatrixDerivative of Matrix<float>
    |FloatDerivative of float

let propagate W b X Y =
    let Y' = computeProbabilityOfSuccess W b X
    let cost = computeCost Y' Y
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

let computeAccuracy (Y':Vector<float>) Y =
    Y' - Y
    |> Vector.map (fun x -> Math.Abs((float)x))
    |> Vector.sum
    |> (fun x -> (1. - x/(float)Y'.Count)*100.)

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

    let accuracy_train = computeAccuracy pred_train (Y_train.Row(0))
    let accuracy_test = computeAccuracy pred_test (Y_test.Row(0))

    printfn "Accuracy of training samples : %A" accuracy_train
    printfn "Accuracy of testing samples : %A" accuracy_test

    trained_params
