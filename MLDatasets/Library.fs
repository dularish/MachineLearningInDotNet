module MLDatasets

open MathNet.Numerics.Data.Text
open OpenCvSharp
open OpenCvSharp
open MathNet.Numerics.LinearAlgebra

let load_catsDatasets = fun () ->
    let train_x = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\train_x.csv", false, ",",true)

    //Matrix X :
    //Always remember to have features in the column direction and samples in the row direction so that the matrix dimension is (n * m) where n is the number of features and m is the samples size

    //Matrix Y :
    //The dimension should be (1*m) where m is the number of samples

    //Input from Andrew Ng course :
    //One common preprocessing step in machine learning is to center and standardize your dataset, 
        //meaning that you substract the mean of the whole numpy array from each example, 
        //and then divide each example by the standard deviation of the whole numpy array. 
        //But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the 
        //dataset by 255 (the maximum value of a pixel channel).

    //During the training of your model, you're going to multiply weights and add biases to some initial inputs in order to observe neuron activations. 
        //Then you backpropogate with the gradients to train the model. 
        //But, it is extremely important for each feature to have a similar range such that our gradients don't explode.

    let train_x_reshapped = train_x.Transpose() / 255.
    let test_x = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\test_x.csv", false, ",",true)
    let test_x_reshapped = test_x.Transpose() / 255.
    let train_y = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\train_y.csv", false, ",",true)
    //let train_y_reshapped = train_y.Row(0)
    let test_y = DelimitedReader.Read<double> (@".\DatasetFiles\CatsDetection\test_y.csv", false, ",",true)
    //let test_y_reshapped = test_y.Row(0)
    train_x_reshapped,test_x_reshapped, train_y, test_y
