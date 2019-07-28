module MLDatasets

open MathNet.Numerics.Data.Text
open OpenCvSharp
open OpenCvSharp
open MathNet.Numerics.LinearAlgebra

let load_catsDatasets = fun () ->
    let train_x = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\train_x.csv", false, ",",true)
    let train_x_reshapped = train_x.Transpose()
    let test_x = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\test_x.csv", false, ",",true)
    let test_x_reshapped = test_x.Transpose()
    let train_y = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\train_y.csv", false, ",",true)
    //let train_y_reshapped = train_y.Row(0)
    let test_y = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\test_y.csv", false, ",",true)
    //let test_y_reshapped = test_y.Row(0)
    train_x_reshapped,test_x_reshapped, train_y, test_y
