#r "../packages/MathNet.Numerics.4.8.1/lib/net461/MathNet.Numerics.dll"
#r "../packages/MathNet.Numerics.FSharp.4.8.1/lib/net45/MathNet.Numerics.FSharp.dll"

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics

let aNumber = 50.

let aRowMatrix = matrix[[1.0;2.0;3.;4.;5.]]

let aColMatrix = aRowMatrix |> Matrix.transpose

let elementWiseMulOfRowMat = aNumber * aRowMatrix
let elementWiseMulOfColMat = aNumber * aColMatrix
let elementWiseAdditionOfRowMat = aNumber + aRowMatrix

printfn "%A" elementWiseMulOfRowMat
printfn "%A" elementWiseMulOfColMat

printfn "Dot Product : %A" (aRowMatrix * (aRowMatrix |> Matrix.transpose))//So in order to do a dot product, they have to be in the format of (rowMatrix * colMatrix)

let aMatrixA = matrix [[ 1.0; 2.0 ]
                       [ 3.0; 4.0 ]]
let aMatrixA' = aMatrixA.Inverse()
let aMatrixB = matrix [[5.;6.]
                       [7.;8.]]
printfn "Inverse of matrixA : %A" aMatrixA'
printfn "Matrix multiplication result : %A" (aMatrixA * aMatrixB)
printfn "Elementwise addition of matrix : %A" (aMatrixA + aNumber)
printfn "Elementwise subtraction of two matrices : %A" (aMatrixB - aMatrixA)
printfn "Elementwise multiplication of two matrices : %A" (aMatrixA .* aMatrixB)