module HelperFunctionsForML

open MathNet.Numerics.LinearAlgebra

let collapseSamples (matrixWithManySamples:Matrix<double>) =
    matrixWithManySamples
    |> Matrix.sumRows
    |> Vector.map (fun x ->
                    (1./(double)(matrixWithManySamples.ColumnCount)) * x)
    |> Matrix<double>.Build.DenseOfColumnVectors//Marking this function for performance improvement possibilities as this creates another memory space to create the matrix