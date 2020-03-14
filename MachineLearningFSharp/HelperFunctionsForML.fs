module HelperFunctionsForML

open MathNet.Numerics.LinearAlgebra
open System.Runtime.Serialization
open System.IO
open MathNet.Numerics
open System.Xml

let collapseSamples (matrixWithManySamples:Matrix<double>) =
    matrixWithManySamples
    |> Matrix.sumRows
    |> Vector.map (fun x ->
                    (1./(double)(matrixWithManySamples.ColumnCount)) * x)
    |> Matrix<double>.Build.DenseOfColumnVectors//Marking this function for performance improvement possibilities as this creates another memory space to create the matrix

let knownTypes = seq{typeof<Matrix<double>>;
                    typeof<LinearAlgebra.Double.DenseMatrix>;
                    typeof<LinearAlgebra.Storage.DenseColumnMajorMatrixStorage<double>>
                    }
    
let serializeModelToFile = fun(model, path) ->
    let serializer = DataContractSerializer(model.GetType(), knownTypes)
        
    let fileWriter = new FileStream(path, FileMode.OpenOrCreate)
    serializer.WriteObject(fileWriter, model)
    fileWriter.Close();
    
let deserializeModelFromFile<'T> path =
    let deserializer = DataContractSerializer(typeof<'T>, knownTypes)
    
    let fileReader = new FileStream(path, FileMode.Open)
    let readerQuotas = XmlDictionaryReaderQuotas()
    readerQuotas.MaxArrayLength <- System.Int32.MaxValue
    let reader = XmlDictionaryReader.CreateTextReader(fileReader, readerQuotas)
    let objRead = deserializer.ReadObject(reader, true)
    reader.Close()
    fileReader.Close()
    match objRead with
    | :? 'T as modelObj -> modelObj
    | _ -> raise (invalidArg "path" "Could not deserialize into desired type of model")