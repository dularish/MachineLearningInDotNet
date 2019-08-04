open MathNet.Numerics.LinearAlgebra
open LogisticRegression
open OpenCvSharp
open OpenCvSharp
open OpenCvSharp
open OpenCvSharp
open OpenCvSharp
open OpenCvSharp
open OpenCVUtils

// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

let showEachImageFromDB = fun(X:Matrix<double>) ->
    for i = 0 to (X.ColumnCount-1) do
        let imageData = X.Column(i).ToArray() |> Array.map (fun(x) -> (byte)x)
        OpenCVUtils.showImageUntilKeyEntered (imageData) ("Image : " + i.ToString()) 3

[<EntryPoint>]
let main argv = 
    
    let train_x,test_x,train_y, test_y = MLDatasets.load_catsDatasets()
    //showEachImageFromDB(train_x)
    //printfn "%A" train_x
    //printfn "%A" train_y
    //printfn "%A" test_x
    //printfn "%A" test_y
    //printfn "%A" initialize_dataset

    let model = NeuralNetwork.createModel train_x train_y test_x test_y 2000 0.005

    System.Console.ReadKey() |> ignore
    printfn "%A" argv
    0 // return an integer exit code
