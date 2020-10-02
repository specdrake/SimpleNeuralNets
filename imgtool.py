from PIL import Image
imgc = Image.open('adit2.jpg')
nimgc = imgc.resize((28,28))
nimgc.save('numtest.jpg')


