from tensorflow.python.keras.models import load_model,model_from_json
#from cv2 import resize,INTER_AREA
from copy import deepcopy
import  cv2
from tensorflow.python.keras.optimizers import Adam


#model2  = load_model('Models/fer_cnn_model_1.h5',compile = False)
#print(model2.summary())

json_file = open('Models/fer_cnn_model_5_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Models/fer_cnn_model_5_weights.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
optimizer = Adam(.001)
#there is issue with compilation ???????????
#loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def processIt(faces = None,img = None):
	face_cascade= cv2.CascadeClassifier('cml/haarcascade_frontalface_default.xml')
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		ret,frame = cap.read()
		frame = cv2.flip(frame,180)
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray)
		for x,y,l,b in faces:
			portion = gray[x:x+l,y:y+b]

			cv2.rectangle(frame,(x,y),(x+l,y+b),(0,0,255),1)
			

			if l > 48:
				resized = cv2.resize(portion,(48,48),interpolation = cv2.INTER_AREA)
			else:
				resized = cv2.resize(portion,(48,48),interpolation = cv2.INTER_LINEAR)
			flat = resized.reshape((1,2304))
			x = loaded_model.predict(flat)
			#print(resized.shape)
			#print(resized.ravel().shape)

			print(x)
		cv2.imshow('win',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

def findEmotion(gray,faces):
	prediction = []
	for x,y,l,b in faces:
		portion = gray[x:x+l,y:y+b]
		if l > 48:
			resized = cv2.resize(portion,(48,48),interpolation = cv2.INTER_AREA)
		else:
			resized = cv2.resize(portion,(48,48),interpolation = cv2.INTER_LINEAR)
		flat = resized.reshape((1,2304))
		x = loaded_model.predict(flat)
		prediction.append(x)
	return prediction
	



#
if __name__ == '__main__':
	print(loaded_model.summary())
	processIt()
