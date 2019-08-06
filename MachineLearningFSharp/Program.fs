open MathNet.Numerics.LinearAlgebra
open LogisticRegression
open OpenCvSharp
open OpenCVUtils
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Timers
open System.Diagnostics

// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

let showEachImageFromDB = fun(X:Matrix<double>) ->
    for i = 0 to (X.ColumnCount-1) do
        let imageData = X.Column(i).ToArray() |> Array.map (fun(x) -> (byte)x)
        OpenCVUtils.showImageUntilKeyEntered (imageData) ("Image : " + i.ToString()) 3

let compareProviderPerformance = fun() ->
    let train_x,test_x,train_y, test_y = MLDatasets.load_catsDatasets()

    
    Control.UseNativeMKL()
    let timer = Stopwatch()
    timer.Start()
    let model = LogisticRegression.createModel train_x train_y test_x test_y 2000 0.0075
    timer.Stop()
    printfn "NativeMKL timer clock : %A" timer.Elapsed.TotalSeconds
    Control.UseManaged()
    
    timer.Restart()
    let model2 = LogisticRegression.createModel train_x train_y test_x test_y 2000 0.0075
    timer.Stop()
    printfn "ManagedProvider timer clock : %A" timer.Elapsed.TotalSeconds



    

[<EntryPoint>]
let main argv = 
    Control.UseNativeMKL()
    printfn "Reading data..."
    let train_x,test_x,train_y, test_y = MLDatasets.load_catsDatasets()
    printfn "Model training started ..."
    //showEachImageFromDB(train_x)
    
    //let model = LogisticRegression.createModel train_x train_y test_x test_y 2000 0.005
    let model = NeuralNetwork.createModel train_x train_y test_x test_y 2500 0.0075

    //compareProviderPerformance()

    System.Console.ReadKey() |> ignore
    printfn "%A" argv
    0 // return an integer exit code
