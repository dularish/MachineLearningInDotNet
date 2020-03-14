namespace MLImplUnitTests

open System
open Microsoft.VisualStudio.TestTools.UnitTesting
open System.IO

[<TestClass>]
type NeuralNetTests () =

    [<TestMethod>]
    member this.AccuracyTestsFor2500Iter0075LearningRate () =
        MathNet.Numerics.Control.UseNativeMKL()
        let train_x,test_x,train_y, test_y = MLDatasets.load_catsDatasets()
        let model = NeuralNetwork.createModel train_x train_y test_x test_y 2500 0.0075
        let pred_train = NeuralNetwork.predictClass train_x model.TrainedParams
        let pred_test = NeuralNetwork.predictClass test_x model.TrainedParams

        let accuracy_train = CommonFunctions.computeAccuracy (pred_train.Row(0)) (train_y.Row(0))
        let accuracy_test = CommonFunctions.computeAccuracy (pred_test.Row(0)) (test_y.Row(0))

        Assert.AreEqual(100.0, accuracy_train)
        Assert.AreEqual(72.0, accuracy_test)

        let path = "Parameters\\testingFile.xml"
        Directory.CreateDirectory(Path.GetDirectoryName(path)) |> ignore

        if File.Exists(path) then File.Delete(path)

        HelperFunctionsForML.serializeModelToFile(model, path)
        Assert.IsTrue(File.Exists(path))

        let deserializedModel = HelperFunctionsForML.deserializeModelFromFile<NeuralNetwork.Model> path
        let pred_train_deserialized = NeuralNetwork.predictClass train_x deserializedModel.TrainedParams
        let pred_test_deserialized = NeuralNetwork.predictClass test_x deserializedModel.TrainedParams

        let accuracy_train_deserialized = CommonFunctions.computeAccuracy (pred_train_deserialized.Row(0)) (train_y.Row(0))
        let accuracy_test_deserialized = CommonFunctions.computeAccuracy (pred_test_deserialized.Row(0)) (test_y.Row(0))

        Assert.AreEqual(accuracy_train, accuracy_train_deserialized)
        Assert.AreEqual(accuracy_test, accuracy_test_deserialized)


