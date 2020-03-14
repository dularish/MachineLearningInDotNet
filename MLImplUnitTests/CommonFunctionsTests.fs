namespace MLImplUnitTests

open Microsoft.VisualStudio.TestTools.UnitTesting
open System.IO

[<TestClass>]
type CommonFunctionsTests () =

    [<TestMethod>]
    member this.ComputeAccuracyReturns100 () =
        let predicted = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0} |> Seq.toArray)
        let actual = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0} |> Seq.toArray)
        let computedAccuracy = CommonFunctions.computeAccuracy predicted actual
        Assert.AreEqual(100.0, computedAccuracy)

    [<TestMethod>]
    member this.ComputeAccuracyReturns0 () =
        let predicted = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0} |> Seq.toArray)
        let actual = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{0.0;1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0;1.0} |> Seq.toArray)
        let computedAccuracy = CommonFunctions.computeAccuracy predicted actual
        Assert.AreEqual(0.0, computedAccuracy)

    [<TestMethod>]
    member this.ComputeAccuracyReturns50Case1 () =
        let predicted = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0} |> Seq.toArray)
        let actual = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0} |> Seq.toArray)
        let computedAccuracy = CommonFunctions.computeAccuracy predicted actual
        Assert.AreEqual(50.0, computedAccuracy)

    [<TestMethod>]
    member this.ComputeAccuracyReturns50Case2 () =
        let predicted = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0;1.0;0.0} |> Seq.toArray)
        let actual = MathNet.Numerics.LinearAlgebra.CreateVector.Dense<float>(seq{0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0} |> Seq.toArray)
        let computedAccuracy = CommonFunctions.computeAccuracy predicted actual
        Assert.AreEqual(50.0, computedAccuracy)