import cv2 as reader

# img = reader.imread(r"C:\Users\ROHIT THORAT\Desktop\hola.jpg", 1)
# reader.imshow("Image bro", img)
# reader.waitKey(10000)
# reader.destroyAllWindows()
# path = "C:\Coding\Python stuff\Python 3.11.0\Hand project"
# reader.imwrite(path + "hola.jpg", img)
# img2 = reader.imread(path + "hola.jpg", 1)
# reader.imshow("Second format", img2)
# reader.waitKey(10000)

img = reader.VideoCapture(0)
img.set(3, 1080)
img.set(4, 720)

# success, frame = img.read()

# if success:
#     reader.imshow("Image", frame)
#     reader.waitKey()
#     reader.destroyAllWindows()

while True:
    flag, frame = img.read()

    if flag:
        reader.imshow("Video", frame)
        if reader.waitKey(1) == ord('q'):
            break

reader.destroyAllWindows()