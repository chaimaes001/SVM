# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:36:14 2021
@author: Chaimae
"""
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# charger la base de donnees des chiffres
chiffres = datasets.load_digits()
""" 
   presentation des donnees utilisees dans le programme pour l implementation du modele SVM,
 donnees d apprentissage, de test et le resultat souhaite 
 on dispose de 1797 echantillon, chaque image est de format 8*8 avec 64 caracteristique,
 chaque caracteristique est represente par un pixel d'image
 """
 
print('Chiffres clés du jeu de données \n{}'.format(chiffres.keys()))
print("l'ensemble de données du resuktat souhaité : \n{}".format(chiffres.target_names))
print('Le format des donnees utilisées : {} \n et les données cibles : {}'.format(chiffres.data.shape,chiffres.target.shape))
print('Le format des images: {}'.format(chiffres.images.shape))

# applatissement des images
n_echanti = len(chiffres.images)
donnees_images = chiffres.images.reshape((n_echanti, -1))

X_train, X_test, y_train, y_test = train_test_split(donnees_images,chiffres.target)
print("La taille des données d'apprentissage et cibles : \n{}, {}".format(X_train.shape,y_train.shape))
print("La taille des données de test et cibles: \n{}, {}".format(X_test.shape,y_test.shape)) 

# Creation du classificateur Machine à vecteurs de supports
svm_clf = svm.SVC(kernel='linear',gamma=0.001)
# lancement de l'entrainement avec les données d'apprentissage
svm_clf.fit(X_train,y_train)
# la phase du test, prediction apres apprentissage
y_pred = svm_clf.predict(X_test)

""" 
Comparons maintenant les valeurs prédites aux valeurs attendues de l'ensemble de données 
 de test à l'aide de sklearn metrics.classification 
"""
 
print("Classification report for classifier %s:\n%s\n"
      % (svm_clf, metrics.classification_report(y_test, y_pred)))

""" 
precision is the fraction of relevant instances among the retrieved instances and is defined as:
precision = tp / (tp + fp) or (true positives)/(prediced positives)

recall is the fraction of relevant instances that have been retrieved over total relevant instances in the image, 
and is defined as
recall = tp / (tp + fn) or (true positives)/(actual positives)
Where, tp = true positives, fp = false positives anf fn = false negatives. 
Recall in this context is also referred to as the true positive rate or sensitivity, 
and precision is also referred to as positive predictive value (PPV).

f1-score: is a measure of a test's accuracy.==> l exactitude du test
It considers both the precision and the recall to compute the score. 
The f1-score can be interpreted as a weighted average of the precision and recall, 
where an f1-score reaches its best value at 1 and worst at 0. 
"""

# print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))

