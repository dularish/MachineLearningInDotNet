module OpenCVUtils

open OpenCvSharp

//The input is a three channel data in which data is provided channel wise that is first red channel data for all the pixel followed by other pixels
let showImageUntilKeyEntered (imageData:byte[]) (windowName:string) (number_of_color_channels:int) =
    let pixels_number = System.Math.Sqrt((double)imageData.Length/(double)number_of_color_channels) |> int
    let mutable someImage = new Mat(pixels_number,pixels_number, MatType.CV_8UC3, imageData)
    someImage <- someImage.CvtColor(ColorConversionCodes.RGB2BGR)//OpenCV needs BGR for display
    Cv2.NamedWindow(windowName, WindowMode.Normal)
    Cv2.ResizeWindow(windowName,600,600)
    Cv2.ImShow(windowName, someImage)
    Cv2.WaitKey(System.Int32.MaxValue) |>ignore
    Cv2.DestroyWindow(windowName)