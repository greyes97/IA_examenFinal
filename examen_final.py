import cv2
import face_recognition as fr
 
mascarilla = fr.load_image_file("./mascarilla.jpg")
sinmascarilla = fr.load_image_file("./sinmascarilla.jpg")
 
con_mascarilla = fr.face_encodings(mascarilla)[0]
sin_mascarilla = fr.face_encodings(sinmascarilla)[0]

encodings_conocidos = [con_mascarilla,sin_mascarilla]

nombres_conocidos = ["Gracias por usar su mascarilla","Use su mascarilla por favor"]

webcam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

reduccion = 4
 
while 1:
    loc_rostros = []
    encodings_rostros = []
    nombres_rostros = []
    nombre = ""
    valido, img = webcam.read()
    if valido:
        img_rgb = img[:, :, ::-1]
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=1.0/reduccion, fy=1.0/reduccion)
        loc_rostros = fr.face_locations(img_rgb)
        encodings_rostros = fr.face_encodings(img_rgb, loc_rostros)

        for encoding in encodings_rostros:
            coincidencias = fr.compare_faces(encodings_rostros, encoding)
            if True in coincidencias:
                nombre = nombres_conocidos[coincidencias.index(True)]
            else:
                nombre = "irreconocible"
            nombres_rostros.append(nombre)
 
        for (top, right, bottom, left), nombre in zip(loc_rostros, nombres_rostros):
            top = top*reduccion
            right = right*reduccion
            bottom = bottom*reduccion
            left = left*reduccion
            if nombre != "irreconocible":
                color = (0,255,0)
            else:
                color = (0,0,255)
 
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.rectangle(img, (left, bottom - 20), (right, bottom), color, -1)
            cv2.putText(img, nombre, (left, bottom - 6), font, 0.6, (0,0,0), 1)
        cv2.imshow('Examen final inteligencia artificial', img)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
 
webcam.release()