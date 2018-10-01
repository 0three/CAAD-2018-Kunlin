# CAAD-2018-Kunlin
Most existing machine learning classifiers are highly vulnerable to adversarial examples. An adversarial example is a sample of input data which has been modified very slightly in a way that is intended to cause a machine learning classifier to misclassify it. In many cases, these modifications can be so subtle that a human observer does not even notice the modification at all, yet the classifier still makes a mistake.

Adversarial examples pose security concerns because they could be used to perform an attack on machine learning systems, even if the adversary has no access to the underlying model.

To accelerate research on adversarial examples, GeekPwn is partnering with Alexey Kurakin from Google Brain and Dawn Song from UC Berkeley EECS to organize Competition on Adversarial Attacks and Defenses 2018 (CAAD2018).

One of sub-competitions in the CAAD is defense against adversarial attack. The goal of defense is to build machine learning classifier which is robust to adversarial example, i.e. can classify adversarial images correctly.

This project proposed a method to defense against adversarial attack. By combining the proposed preprocessing method with an adversarially trained model, it ranked No.5 in the CAAD2018 defense sub-competition(https://en.caad.geekpwn.org/competition/list.html?menuId=10)

# The approach

The main ideal of the defense is to utilize preprocessing to defend adversarial examples:
* Adding gaussian blur: 
```python
image = cv2.GaussianBlur(image,(7,7),5)
```
* Weiner filtering:
```python
image = wiener(image) / 255.0
```
 ![image](https://github.com/0three/CAAD-2018-Kunlin/blob/master/CAAD-kunlin.png)
 
 # Pros
 
 * 1.Easy to reappearence and extend. 
 * 2.Compatiable to popular image classification networks. (i.e., we use preprocessing + adversarial training + Rexnet_V2_50 in our submission)

# Team Member

* Kunlin Liu (University of Science and Technology in China)
* Xiaoyu Ye (Sensetime)

# Leaderboard
Our team name is Kunlin, and our rank is No.5.(https://en.caad.geekpwn.org/competition/list.html?menuId=10)
