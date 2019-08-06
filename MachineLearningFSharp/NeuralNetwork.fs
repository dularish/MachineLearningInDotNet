module NeuralNetwork

open MathNet.Numerics.LinearAlgebra
open System.Collections.Generic
open CommonFunctions
open HelperFunctionsForML
open MathNet.Numerics.Random
open MathNet.Numerics

exception IncorrectLayerData of string

type InputCache(weights, bias, inputForLayer, ZComputed) =
    member this.W = weights
    member this.b = bias
    member this.InputForLayer = inputForLayer
    member this.Z = ZComputed

type LayerParameters(weights, bias) =
    member this.W = weights
    member this.b = bias

type BackPropCache(dW, db) =
    member this.dW = dW
    member this.db = db

let initializeParameters (layerDims: int list) =
    if layerDims.Length < 3 then
        raise (IncorrectLayerData "Function : Initialize parameters. Number of layers must be atleast 3")
    elif layerDims.[layerDims.Length - 1] <> 1 then
        raise (IncorrectLayerData "Function : Intialize parameters. The last layer must have a dimension of 1")
    else
        let mutable parameters = Dictionary<int, LayerParameters>()
        [1..(layerDims.Length-1)]
        |> List.iter (fun (i) ->
                        let Weights = 
                            (Matrix<double>.Build.Random(layerDims.[i], layerDims.[i-1])) / (sqrt ((double)layerDims.[i-1]))
                        let bias = 
                            ((double)0.01) * (Matrix<double>.Build.Dense(layerDims.[i], 1, 0.))
                        parameters.Add(i, LayerParameters(Weights, bias))
                     )
        parameters

let forwardPropagate (inputForLayer:Matrix<double>) (layerParameters:LayerParameters) (forwardPropagateFunc) =
    let Z = 
        (layerParameters.W * inputForLayer)
        |> Matrix.mapCols (fun (index) (columnMatrix) ->
                                let parameterToAdd = layerParameters.b.Column(0)
                                (columnMatrix + parameterToAdd)
                                )
    let activated = 
        Z
        |> Matrix.map forwardPropagateFunc

    activated, InputCache(layerParameters.W, layerParameters.b, inputForLayer, Z)

let forwardPropagateAllLayers (parameters:Dictionary<int, LayerParameters>) (inputMatrix:Matrix<double>) =
    let layers = 
        parameters.Keys
        |> Seq.sort
        |> Seq.toList
    
    let mutable previousLayerOutput = inputMatrix
    let cachesDict = Dictionary<int, InputCache>()

    layers
    |> Seq.iter (fun (x) ->
                    let A, inputCache = forwardPropagate (previousLayerOutput) (parameters.[x]) (if x = layers.[layers.Length - 1] then sigmoid else relu)
                    cachesDict.Add(x, inputCache)
                    previousLayerOutput <- A
                    )

    previousLayerOutput, cachesDict
                
let backwardPropagate (dA:Matrix<double>) (inputCache:InputCache) (backDerivativeFn) =
    let dZ = backDerivativeFn dA inputCache.Z

    let dW = 
        dZ * (inputCache.InputForLayer.Transpose())
        |> Matrix.map (fun (x) ->
                            (1./(double)inputCache.InputForLayer.ColumnCount) * x)
    let db = 
        dZ
        |> collapseSamples
    let dX =
        inputCache.W.Transpose() * dZ

    dX , BackPropCache(dW, db)

let backPropagateAllLayers (Y':Matrix<float>) (Y:Matrix<float>) (inputCaches:Dictionary<int, InputCache>) =
    
    let mutable dANextLayer = 
        - ((Y ./ Y') - ((1. - Y) ./ (1. - Y')))

    let layers = 
        inputCaches.Keys
        |> Seq.sort
        |> Seq.rev
        |> Seq.toList

    let gradsDict = Dictionary<int, BackPropCache>()

    layers
    |> Seq.iter (fun (x) ->
                    let dX, backPropCache = backwardPropagate (dANextLayer) (inputCaches.[x]) (if x = layers.[0] then dZSigmoidBackDerivative else dZReluBackDerivative)
                    gradsDict.Add(x, backPropCache)
                    dANextLayer <- dX)

    gradsDict

let updateParameters (parameters:Dictionary<int, LayerParameters>) (backPropGrads:Dictionary<int, BackPropCache>) (learningRate:double) =
    
    let updatedParams = Dictionary<int, LayerParameters>()
    let layers =
        parameters.Keys
        |> Seq.iter(fun (x) ->
                        let backPropCache = backPropGrads.[x]
                        let inputParams = parameters.[x]
                        let updatedW = inputParams.W - (learningRate * backPropCache.dW)
                        let updated_b = inputParams.b - (learningRate * backPropCache.db)
                        updatedParams.Add(x, LayerParameters(updatedW, updated_b))
                        )
    updatedParams

let predictClass X parameters =
    let Y', _ = forwardPropagateAllLayers parameters X
    Y'

let createModel (X_train:Matrix<double>) Y_train X_test (Y_test:Matrix<float>) (num_iterations) learning_rate =
    let mutable parameters = initializeParameters [X_train.RowCount; 20; 7; 5; 1]

    [1..num_iterations]
    |> List.iter(fun(i) ->
                    let Y', inputCaches = forwardPropagateAllLayers parameters X_train

                    let cost = computeCost Y' Y_train

                    if i%100 = 0 then
                        printfn "Cost after %A iterations : %A" i cost
    
                    let grads = backPropagateAllLayers Y' Y_train inputCaches

                    let updatedParams = updateParameters parameters grads learning_rate
                    parameters <- updatedParams
                    )

    let pred_train = predictClass X_train parameters
    let pred_test = predictClass X_test parameters

    let accuracy_train = CommonFunctions.computeAccuracy (pred_train.Row(0)) (Y_train.Row(0))
    let accuracy_test = CommonFunctions.computeAccuracy (pred_test.Row(0)) (Y_test.Row(0))

    printfn "Accuracy of training samples : %A" accuracy_train
    printfn "Accuracy of testing samples : %A" accuracy_test

    
