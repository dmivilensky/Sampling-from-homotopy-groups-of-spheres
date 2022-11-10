# Статистика генерирования методами с фильтрацией

$n$ — число образующих, $l$ — максимальная длина генерируемого слова

## Случайное сэмплирование
### $n = 2, l = 25$

* 1000 слов из $\langle x\rangle$ за 0.95 с
  
    Примеры:

    - `y⁻¹xyx⁻¹y⁻¹xy⁻¹x⁻¹yx⁻¹yxy⁻¹x⁻¹yyx⁻¹y⁻¹xyxy⁻¹x⁻¹x⁻¹x⁻¹`
    - `x⁻¹yx⁻¹y⁻¹y⁻¹x⁻¹y⁻¹x⁻¹yyxy⁻¹y⁻¹xyxyyxy⁻¹xy⁻¹x⁻¹y`
    - `yxxy⁻¹y⁻¹y⁻¹y⁻¹xyx⁻¹y⁻¹x⁻¹yxy⁻¹x⁻¹yyyyx⁻¹y⁻¹y⁻¹x⁻¹y`
    - `y⁻¹y⁻¹xyyyxyyx⁻¹yx⁻¹y⁻¹xy⁻¹y⁻¹x⁻¹y⁻¹y⁻¹y⁻¹x⁻¹yyx`
    - `xy⁻¹xyxyxy⁻¹xy⁻¹x⁻¹yx⁻¹yx⁻¹y⁻¹x⁻¹y⁻¹x⁻¹yx⁻¹yxy⁻¹x`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle$ за 4.5 c

    Примеры: 

    - `y⁻¹xxxy⁻¹x⁻¹yx⁻¹x⁻¹x⁻¹yxyx⁻¹y⁻¹xyxy⁻¹xy⁻¹x⁻¹yx⁻¹`
    - `x⁻¹yyyyx⁻¹y⁻¹y⁻¹y⁻¹xyyyxy⁻¹y⁻¹y⁻¹x⁻¹y⁻¹xy⁻¹xyx⁻¹`
    - `xyyx⁻¹y⁻¹x⁻¹x⁻¹y⁻¹y⁻¹xyyxxyxy⁻¹y⁻¹x⁻¹y⁻¹xyx⁻¹x⁻¹`
    - `yx⁻¹y⁻¹y⁻¹xy⁻¹y⁻¹x⁻¹yyx⁻¹y⁻¹y⁻¹xyyx⁻¹yyxy⁻¹y⁻¹xy`
    - `y⁻¹x⁻¹yxxy⁻¹y⁻¹y⁻¹x⁻¹x⁻¹yxy⁻¹xxyyyx⁻¹x⁻¹yx⁻¹y⁻¹x`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle xy\rangle$ за 2.6 c

    Примеры:

    - `yyx⁻¹yyxyx⁻¹yx⁻¹x⁻¹y⁻¹xy⁻¹xy⁻¹x⁻¹y⁻¹y⁻¹xy⁻¹x`
    - `y⁻¹xxy⁻¹x⁻¹yyyxy⁻¹x⁻¹x⁻¹y⁻¹y⁻¹y⁻¹xyx⁻¹x⁻¹yxxyx⁻¹`
    - `y⁻¹y⁻¹x⁻¹y⁻¹y⁻¹y⁻¹x⁻¹yyxyyx⁻¹yyxxyx⁻¹y⁻¹y⁻¹x`
    - `yx⁻¹yyxxy⁻¹x⁻¹x⁻¹yyxy⁻¹xxyx⁻¹x⁻¹y⁻¹y⁻¹xy⁻¹x⁻¹y⁻¹`
    - `xy⁻¹xy⁻¹xy⁻¹xy⁻¹xyyx⁻¹yx⁻¹yx⁻¹x⁻¹x⁻¹x⁻¹y⁻¹y⁻¹xyy`

### $n = 3, l = 15$

* 1000 слов из $\langle x\rangle$ за 0.5 с

    Примеры:

    - `x⁻¹y⁻¹z⁻¹x⁻¹yx⁻¹y⁻¹xzyxy⁻¹xy`
    - `z⁻¹xzx⁻¹z⁻¹x⁻¹zyxy⁻¹z⁻¹x⁻¹zx⁻¹`
    - `xy⁻¹y⁻¹x⁻¹yyx⁻¹z⁻¹xy⁻¹x⁻¹yx⁻¹zx`
    - `y⁻¹xxyxy⁻¹x⁻¹x⁻¹yxzxz⁻¹x`
    - `x⁻¹y⁻¹y⁻¹x⁻¹yyz⁻¹x⁻¹zyx⁻¹y⁻¹zx⁻¹z⁻¹`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle$ за 20 c

    Примеры: 

    - `yxzyz⁻¹x⁻¹zy⁻¹z⁻¹x⁻¹y⁻¹y⁻¹xy`
    - `yyyx⁻¹y⁻¹y⁻¹xyyxy⁻¹y⁻¹y⁻¹x⁻¹`
    - `xyx⁻¹zy⁻¹x⁻¹yz⁻¹xy⁻¹x⁻¹zxz⁻¹`
    - `z⁻¹xy⁻¹zyxy⁻¹z⁻¹yzx⁻¹z⁻¹x⁻¹z`
    - `yzy⁻¹z⁻¹xzyz⁻¹xy⁻¹y⁻¹x⁻¹yx⁻¹`

* 50 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$ за 16 с

    Примеры:

    - `x⁻¹x⁻¹z⁻¹yzy⁻¹x⁻¹yz⁻¹y⁻¹zxxx`
    - `y⁻¹zyz⁻¹z⁻¹xzzy⁻¹z⁻¹yz⁻¹x⁻¹z`
    - `z⁻¹yzy⁻¹zx⁻¹z⁻¹yz⁻¹y⁻¹zzxz⁻¹`
    - `z⁻¹z⁻¹y⁻¹zzxz⁻¹z⁻¹yzzy⁻¹x⁻¹y`
    - `zyz⁻¹z⁻¹y⁻¹x⁻¹yzzy⁻¹z⁻¹z⁻¹xz`

* 1 слово из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle xyz\rangle$ за 3050 c

    Примеры:

    - `z⁻¹yzy⁻¹x⁻¹z⁻¹xzyz⁻¹y⁻¹x⁻¹zx`

### $n = 4, l = 30$

* 1000 слов из $\langle x\rangle$ за 0.8 с

    Примеры:

    - `p⁻¹x⁻¹p⁻¹xyz⁻¹p⁻¹y⁻¹zx⁻¹y⁻¹pxp⁻¹yxz⁻¹ypzy⁻¹x⁻¹pxpyx⁻¹y⁻¹`
    - `ypz⁻¹y⁻¹pzzyxz⁻¹y⁻¹z⁻¹pxp⁻¹zyzx⁻¹y⁻¹z⁻¹z⁻¹p⁻¹yzp⁻¹y⁻¹zxz⁻¹`
    - `x⁻¹y⁻¹p⁻¹p⁻¹y⁻¹y⁻¹xzxp⁻¹y⁻¹px⁻¹p⁻¹ypx⁻¹z⁻¹x⁻¹yyppyxzx⁻¹z⁻¹`
    - `p⁻¹x⁻¹yz⁻¹x⁻¹z⁻¹x⁻¹ppx⁻¹x⁻¹z⁻¹p⁻¹z⁻¹xzpzxxp⁻¹p⁻¹xzxzy⁻¹xpx`
    - `z⁻¹xxxz⁻¹x⁻¹ppzx⁻¹zp⁻¹yx⁻¹y⁻¹pz⁻¹xz⁻¹p⁻¹p⁻¹xzx⁻¹x⁻¹x⁻¹x⁻¹z`

* 50 слов из $\langle x\rangle \cap \langle y\rangle$ за 50 c

    Примеры:

    - `py⁻¹p⁻¹z⁻¹yzxyx⁻¹zy⁻¹x⁻¹yz⁻¹xy⁻¹x⁻¹z⁻¹y⁻¹zpyp⁻¹zy⁻¹xyz⁻¹`
    - `x⁻¹zxz⁻¹yyyzyyyx⁻¹z⁻¹x⁻¹zxy⁻¹y⁻¹y⁻¹z⁻¹y⁻¹y⁻¹y⁻¹zx⁻¹z⁻¹xx`
    - `y⁻¹xzxypyp⁻¹x⁻¹y⁻¹z⁻¹x⁻¹y⁻¹x⁻¹yxzyxpy⁻¹p⁻¹y⁻¹x⁻¹z⁻¹x⁻¹yx`
    - `yppz⁻¹pyp⁻¹zp⁻¹p⁻¹yyyx⁻¹y⁻¹y⁻¹y⁻¹ppz⁻¹py⁻¹p⁻¹zp⁻¹p⁻¹y⁻¹x`
    - `z⁻¹xz⁻¹xy⁻¹x⁻¹yzyx⁻¹zy⁻¹x⁻¹yz⁻¹xy⁻¹z⁻¹y⁻¹xyx⁻¹zx⁻¹zy⁻¹xy`

* 2 слова из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$ за 1100 c

    Примеры:

    - `zy⁻¹y⁻¹z⁻¹x⁻¹y⁻¹zyyz⁻¹y⁻¹xyxy⁻¹x⁻¹yzy⁻¹y⁻¹z⁻¹yxzyyz⁻¹y⁻¹x⁻¹y`
    - `y⁻¹xz⁻¹x⁻¹z⁻¹yzxy⁻¹x⁻¹yx⁻¹z⁻¹y⁻¹zxzx⁻¹yxz⁻¹xzx⁻¹y⁻¹xyx⁻¹`

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle p\rangle \cap \langle xyzp\rangle$


### $n = 5, l = 60$

* 1000 слов из $\langle x\rangle$ за 1.2 с

    Примеры:
    
    - `qyp⁻¹xy⁻¹z⁻¹q⁻¹zx⁻¹zxqz⁻¹p⁻¹x⁻¹ypyxy⁻¹x⁻¹p⁻¹qxpq⁻¹q⁻¹px⁻¹p⁻¹qqp⁻¹x⁻¹q⁻¹pxyx⁻¹y⁻¹p⁻¹y⁻¹xpzq⁻¹x⁻¹z⁻¹xz⁻¹qzyx⁻¹py⁻¹q⁻¹pxp⁻¹`
    - `zq⁻¹zp⁻¹x⁻¹p⁻¹yp⁻¹p⁻¹yzxzxzpz⁻¹qyqp⁻¹p⁻¹z⁻¹xypq⁻¹zx⁻¹z⁻¹qp⁻¹y⁻¹x⁻¹zppq⁻¹y⁻¹q⁻¹zp⁻¹z⁻¹x⁻¹z⁻¹x⁻¹z⁻¹y⁻¹ppy⁻¹pxpz⁻¹qz⁻¹qx⁻¹q⁻¹`
    - `qzqz⁻¹p⁻¹p⁻¹p⁻¹y⁻¹q⁻¹p⁻¹q⁻¹xxq⁻¹z⁻¹ypq⁻¹pzp⁻¹yq⁻¹p⁻¹zq⁻¹q⁻¹yxy⁻¹qqz⁻¹pqy⁻¹pz⁻¹p⁻¹qp⁻¹y⁻¹zqx⁻¹x⁻¹qpqypppzq⁻¹z⁻¹q⁻¹yx⁻¹y⁻¹`
    - `xyzypy⁻¹x⁻¹y⁻¹p⁻¹y⁻¹q⁻¹y⁻¹x⁻¹z⁻¹x⁻¹q⁻¹p⁻¹zxyyq⁻¹p⁻¹qxy⁻¹q⁻¹yx⁻¹y⁻¹qyx⁻¹q⁻¹pqy⁻¹y⁻¹x⁻¹z⁻¹pqxzxyqypyxyp⁻¹y⁻¹z⁻¹y⁻¹x⁻¹p⁻¹xp`
    - `qz⁻¹y⁻¹z⁻¹xz⁻¹z⁻¹p⁻¹xq⁻¹xzq⁻¹xpy⁻¹xz⁻¹xpx⁻¹q⁻¹x⁻¹p⁻¹xqzyp⁻¹x⁻¹py⁻¹z⁻¹q⁻¹x⁻¹pxqxp⁻¹x⁻¹zx⁻¹yp⁻¹x⁻¹qz⁻¹x⁻¹qx⁻¹pzzx⁻¹zyzxq⁻¹`

* 0 слов из $\langle x\rangle \cap \langle y\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle p\rangle \cap \langle q\rangle \cap \langle xyzpq\rangle$


## Эволюционный метод оптимизации длины редуцированного слова после подстановки
### $n = 2, l = 25$
* 1000 слов из $\langle x\rangle$ за 9.2 с
  
    Примеры:
    - `y⁻¹xy⁻¹xxy⁻¹xy⁻¹xyx⁻¹yx⁻¹x⁻¹yx⁻¹yyx⁻¹y⁻¹y⁻¹xy`
    - `xy⁻¹y⁻¹y⁻¹y⁻¹x⁻¹x⁻¹yyxy⁻¹y⁻¹xxyyyyx⁻¹y⁻¹xyyxy⁻¹`
    - `y⁻¹xxyx⁻¹yxxy⁻¹x⁻¹yx⁻¹x⁻¹y⁻¹xy⁻¹x⁻¹x⁻¹x⁻¹yx⁻¹y⁻¹x⁻¹y`
    - `x⁻¹x⁻¹x⁻¹y⁻¹xyyx⁻¹yxy⁻¹xy⁻¹y⁻¹x⁻¹yxxxy⁻¹xyx⁻¹`
    - `x⁻¹x⁻¹yx⁻¹y⁻¹y⁻¹xyx⁻¹y⁻¹x⁻¹yxy⁻¹x⁻¹yyxy⁻¹y⁻¹xyyx⁻¹y⁻¹`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle$ за 2.3 c

    Примеры: 

    - `yyxy⁻¹y⁻¹y⁻¹x⁻¹y`
    - `y⁻¹y⁻¹x⁻¹yx⁻¹x⁻¹y⁻¹x⁻¹yxy⁻¹x⁻¹yx⁻¹y⁻¹xxyxxy⁻¹xyy`
    - `y⁻¹y⁻¹x⁻¹yyxy⁻¹xyx⁻¹`
    - `yx⁻¹yx⁻¹x⁻¹yxy⁻¹xy⁻¹y⁻¹y⁻¹x⁻¹yyxy⁻¹x`
    - `xy⁻¹xy⁻¹x⁻¹yx⁻¹x⁻¹y⁻¹x⁻¹yxxy⁻¹xyx⁻¹yx⁻¹yx⁻¹y⁻¹xx`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle xy\rangle$ за 43.5 c

    Примеры: 

    - `y⁻¹xyx⁻¹y⁻¹xyxyx⁻¹y⁻¹y⁻¹x⁻¹y⁻¹x⁻¹yxy⁻¹x⁻¹yyx`
    - `x⁻¹x⁻¹x⁻¹x⁻¹x⁻¹y⁻¹y⁻¹y⁻¹x⁻¹y⁻¹xxyx⁻¹yxyyyxxxxy⁻¹`
    - `yxy⁻¹xyyyx⁻¹y⁻¹y⁻¹x⁻¹yxy⁻¹y⁻¹y⁻¹x⁻¹yx⁻¹y⁻¹y⁻¹xyy`
    - `yxy⁻¹xy⁻¹xxyxy⁻¹xyyx⁻¹y⁻¹x⁻¹x⁻¹yx⁻¹yx⁻¹y⁻¹x⁻¹y⁻¹`
    - `y⁻¹x⁻¹yx⁻¹yyxxxxyx⁻¹x⁻¹x⁻¹y⁻¹y⁻¹xy⁻¹xyx⁻¹y⁻¹`

### $n = 3, l = 15$
* 1000 слов из $\langle x\rangle$ за 6.2 с
  
    Примеры:
    - `xz⁻¹yyxz⁻¹xzx⁻¹y⁻¹y⁻¹zx⁻¹`
    - `y⁻¹x⁻¹zxzx⁻¹z⁻¹x⁻¹z⁻¹xyyxy⁻¹`
    - `yzy⁻¹xyx⁻¹y⁻¹x⁻¹yz⁻¹y⁻¹x⁻¹y⁻¹x⁻¹y`
    - `yzy⁻¹z⁻¹y⁻¹x⁻¹yzyz⁻¹y⁻¹y⁻¹xyx⁻¹`
    - `z⁻¹xzxy⁻¹z⁻¹x⁻¹zyx⁻¹z⁻¹x⁻¹z`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle$ за 5.5 c

    Примеры: 

    - `yx⁻¹y⁻¹x`
    - `xyx⁻¹y⁻¹x⁻¹yxy⁻¹x⁻¹y⁻¹xy`
    - `xyxyx⁻¹y⁻¹x⁻¹y⁻¹z⁻¹y⁻¹x⁻¹yxz`
    - `xy⁻¹y⁻¹x⁻¹y⁻¹xyx⁻¹yy`
    - `yz⁻¹xyx⁻¹y⁻¹zx⁻¹y⁻¹x`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$ за 59 c

    Примеры: 

    - `yx⁻¹zy⁻¹zyxy⁻¹x⁻¹z⁻¹yz⁻¹y⁻¹x`
    - `y⁻¹z⁻¹yzxz⁻¹y⁻¹zyx⁻¹`
    - `x⁻¹z⁻¹xyz⁻¹z⁻¹y⁻¹x⁻¹yxzzx⁻¹y⁻¹zx`
    - `zxz⁻¹x⁻¹y⁻¹y⁻¹xzx⁻¹z⁻¹yy`
    - `y⁻¹zyz⁻¹xzy⁻¹z⁻¹yx⁻¹`

* 4 слова из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle xyz\rangle$ за 363 c

    Примеры: 

    - `z⁻¹xy⁻¹x⁻¹yzxz⁻¹x⁻¹y⁻¹xyzx⁻¹`
    - `yx⁻¹zxz⁻¹y⁻¹x⁻¹yxzx⁻¹z⁻¹y⁻¹x`
    - `yz⁻¹y⁻¹x⁻¹zxz⁻¹yzy⁻¹x⁻¹z⁻¹xz`
    - `zxyx⁻¹y⁻¹z⁻¹yzxy⁻¹x⁻¹yz⁻¹y⁻¹`

### $n = 4, l = 30$
* 1000 слов из $\langle x\rangle$ за 17 с
  
    Примеры:
    - `z⁻¹xyyxp⁻¹zxyx⁻¹zpy⁻¹x⁻¹yp⁻¹z⁻¹xy⁻¹x⁻¹z⁻¹px⁻¹y⁻¹y⁻¹x⁻¹zyxy⁻¹`
    - `xzpxxz⁻¹z⁻¹x⁻¹yzpz⁻¹xzp⁻¹z⁻¹y⁻¹xzzx⁻¹x⁻¹p⁻¹z⁻¹x⁻¹p⁻¹xpx⁻¹`
    - `yyx⁻¹py⁻¹zzzx⁻¹zypx⁻¹px⁻¹p⁻¹xp⁻¹y⁻¹z⁻¹xz⁻¹z⁻¹z⁻¹yp⁻¹xy⁻¹y⁻¹`
    - `yp⁻¹y⁻¹z⁻¹yz⁻¹y⁻¹pz⁻¹y⁻¹xz⁻¹xzx⁻¹yzp⁻¹yzy⁻¹zypy⁻¹x⁻¹y⁻¹xy`
    - `x⁻¹x⁻¹zzzpy⁻¹zxy⁻¹z⁻¹pxp⁻¹zyx⁻¹z⁻¹yp⁻¹z⁻¹z⁻¹z⁻¹xxyx⁻¹y⁻¹`

* 1000 слов из $\langle x\rangle \cap \langle y\rangle$ за 30.7 c

    Примеры: 

    - `x⁻¹p⁻¹x⁻¹y⁻¹xypx`
    - `zzpyp⁻¹zxz⁻¹yz⁻¹p⁻¹xy⁻¹x⁻¹ypzy⁻¹zx⁻¹z⁻¹py⁻¹p⁻¹z⁻¹z⁻¹`
    - `x⁻¹y⁻¹p⁻¹yz⁻¹yz⁻¹p⁻¹ypyxy⁻¹p⁻¹y⁻¹px⁻¹zy⁻¹zy⁻¹pxy`
    - `y⁻¹p⁻¹xy⁻¹x⁻¹y⁻¹xy⁻¹xy⁻¹x⁻¹p⁻¹y⁻¹x⁻¹yxpyx⁻¹yx⁻¹yxypy`
    - `p⁻¹z⁻¹p⁻¹y⁻¹zy⁻¹x⁻¹p⁻¹p⁻¹zzyx⁻¹y⁻¹xz⁻¹z⁻¹ppxyz⁻¹ypzp`

* 50 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$ за 94 c

    Примеры: 

    - `zy⁻¹z⁻¹yp⁻¹x⁻¹py⁻¹zyz⁻¹p⁻¹xp`
    - `xy⁻¹x⁻¹z⁻¹xzyz⁻¹x⁻¹z`
    - `xy⁻¹y⁻¹pppzy⁻¹z⁻¹yp⁻¹xpy⁻¹zyz⁻¹p⁻¹x⁻¹p⁻¹p⁻¹yyx⁻¹`
    - `x⁻¹z⁻¹yzxz⁻¹x⁻¹y⁻¹xz`
    - `yz⁻¹z⁻¹y⁻¹p⁻¹z⁻¹x⁻¹y⁻¹y⁻¹z⁻¹z⁻¹yzyzxz⁻¹y⁻¹z⁻¹y⁻¹zzyyzpyzzy⁻¹`

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle p\rangle \cap \langle xyzp\rangle$

### $n = 5, l = 60$
* 1000 слов из $\langle x\rangle$ за 28 с
  
    Примеры:
    - `z⁻¹y⁻¹xqz⁻¹qp⁻¹y⁻¹z⁻¹p⁻¹x⁻¹zp⁻¹zxqqy⁻¹qxxypq⁻¹p⁻¹q⁻¹pz⁻¹xzp⁻¹qpqp⁻¹y⁻¹x⁻¹x⁻¹q⁻¹yq⁻¹q⁻¹x⁻¹z⁻¹pz⁻¹xpzypq⁻¹zq⁻¹x⁻¹yzqxq⁻¹`
    - `zx⁻¹zyz⁻¹x⁻¹q⁻¹y⁻¹z⁻¹z⁻¹y⁻¹xyp⁻¹xq⁻¹z⁻¹pq⁻¹x⁻¹px⁻¹pq⁻¹yyqxq⁻¹y⁻¹y⁻¹qp⁻¹xp⁻¹xqp⁻¹zqx⁻¹py⁻¹x⁻¹yzzyqxzy⁻¹z⁻¹xz⁻¹y⁻¹x⁻¹y`
    - `yp⁻¹qqzyx⁻¹pqp⁻¹x⁻¹p⁻¹p⁻¹y⁻¹xz⁻¹xqx⁻¹y⁻¹pzpq⁻¹xqx⁻¹y⁻¹p⁻¹x⁻¹pyxq⁻¹x⁻¹qp⁻¹z⁻¹p⁻¹yxq⁻¹x⁻¹zx⁻¹yppxpq⁻¹p⁻¹xy⁻¹z⁻¹q⁻¹q⁻¹py⁻¹`
    - `p⁻¹p⁻¹q⁻¹y⁻¹z⁻¹z⁻¹q⁻¹q⁻¹p⁻¹p⁻¹xq⁻¹xpyp⁻¹x⁻¹y⁻¹pxyzyq⁻¹p⁻¹yq⁻¹p⁻¹xpqy⁻¹pqy⁻¹z⁻¹y⁻¹x⁻¹p⁻¹yxpy⁻¹p⁻¹x⁻¹qx⁻¹ppqqzzyqppyx⁻¹y⁻¹`
    - `y⁻¹pqypyx⁻¹y⁻¹p⁻¹p⁻¹x⁻¹yppyz⁻¹y⁻¹y⁻¹qpzyp⁻¹y⁻¹zy⁻¹xzx⁻¹z⁻¹x⁻¹yz⁻¹ypy⁻¹z⁻¹p⁻¹q⁻¹yyzy⁻¹p⁻¹p⁻¹y⁻¹xppyxy⁻¹p⁻¹y⁻¹q⁻¹p⁻¹yp⁻¹xp`

* 50 слов из $\langle x\rangle \cap \langle y\rangle$ за 25.7 c

    Примеры: 

    - `y⁻¹xyyx⁻¹y⁻¹`
    - `xqzq⁻¹y⁻¹y⁻¹q⁻¹pppyqqy⁻¹xy⁻¹zq⁻¹p⁻¹p⁻¹y⁻¹qzpxz⁻¹p⁻¹zyx⁻¹y⁻¹xz⁻¹pzx⁻¹p⁻¹z⁻¹q⁻¹yppqz⁻¹yx⁻¹yq⁻¹q⁻¹y⁻¹p⁻¹p⁻¹p⁻¹qyyqz⁻¹q⁻¹x⁻¹`
    - `z⁻¹yzzxy⁻¹x⁻¹yx⁻¹z⁻¹z⁻¹y⁻¹zzxz⁻¹`
    - `y⁻¹zzqz⁻¹qqyyp⁻¹zyp⁻¹z⁻¹xxyxz⁻¹p⁻¹q⁻¹yzqzx⁻¹zy⁻¹z⁻¹y⁻¹xyzyz⁻¹z⁻¹q⁻¹z⁻¹y⁻¹qpzx⁻¹y⁻¹x⁻¹x⁻¹zpy⁻¹z⁻¹py⁻¹y⁻¹q⁻¹q⁻¹zq⁻¹z⁻¹z⁻¹y`
    - `q⁻¹y⁻¹y⁻¹zp⁻¹y⁻¹z⁻¹q⁻¹xy⁻¹z⁻¹p⁻¹z⁻¹xypxy⁻¹z⁻¹pz⁻¹pz⁻¹y⁻¹z⁻¹y⁻¹xyx⁻¹zyzp⁻¹zp⁻¹zyx⁻¹p⁻¹y⁻¹x⁻¹zpzyx⁻¹qzypz⁻¹yyq`

* 4 слова из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$ за 633 c

    Примеры: 

    - `zyz⁻¹xzx⁻¹y⁻¹xz⁻¹x⁻¹`
    - `yz⁻¹y⁻¹xyzy⁻¹z⁻¹x⁻¹z`
    - `xzp⁻¹p⁻¹x⁻¹y⁻¹q⁻¹zyxp⁻¹xp⁻¹zp⁻¹z⁻¹pz⁻¹y⁻¹p⁻¹q⁻¹z⁻¹p⁻¹xzyz⁻¹y⁻¹x⁻¹yzy⁻¹z⁻¹pzqpyzp⁻¹zpz⁻¹px⁻¹px⁻¹y⁻¹z⁻¹qyxppz⁻¹x⁻¹`
    - `zyz⁻¹y⁻¹xyzy⁻¹z⁻¹x⁻¹`

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle p\rangle \cap \langle q\rangle \cap \langle xyzpq\rangle$


## Градиентный метод оптимизации расстояния в представлениях Санова
### $n = 2, l = 25$
* 100 слов из $\langle x\rangle$ за 17.2 с
  
    Примеры:
    - `yx⁻¹y⁻¹x⁻¹y⁻¹x⁻¹yyyx⁻¹x⁻¹y⁻¹x⁻¹y⁻¹`
    - `yx⁻¹yyx⁻¹x⁻¹x⁻¹y⁻¹x⁻¹y⁻¹x⁻¹yx⁻¹y⁻¹y⁻¹`
    - `yx⁻¹y⁻¹y⁻¹xxyyxy⁻¹y⁻¹x⁻¹y`
    - `yx⁻¹x⁻¹y⁻¹y⁻¹y⁻¹x⁻¹x⁻¹yyx⁻¹x⁻¹x⁻¹`
    - `yxy⁻¹y⁻¹x⁻¹yyxy⁻¹xy⁻¹y⁻¹x⁻¹yyx⁻¹x⁻¹`

* 100 слов из $\langle x\rangle \cap \langle y\rangle$ за 72 c

    Примеры: 

    - `x⁻¹yxxxy⁻¹x⁻¹yxy⁻¹xy⁻¹x⁻¹yx⁻¹yyx⁻¹y⁻¹y⁻¹`
    - `yx⁻¹yyx⁻¹x⁻¹y⁻¹xxxy⁻¹xxy⁻¹x⁻¹x⁻¹`
    - `yx⁻¹y⁻¹y⁻¹xxyx⁻¹y⁻¹x⁻¹y⁻¹xyy`
    - `yx⁻¹yyx⁻¹x⁻¹x⁻¹y⁻¹x⁻¹y⁻¹x⁻¹yx⁻¹y⁻¹y⁻¹`
    - `y⁻¹xxy⁻¹xyx⁻¹y⁻¹y⁻¹x⁻¹x⁻¹x⁻¹yyxyyyxy⁻¹y⁻¹`

* 100 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle xy\rangle$ за 95 c

    Примеры: 

    - `y⁻¹xyyx⁻¹y⁻¹y⁻¹xxy⁻¹x⁻¹x⁻¹yy`
    - `y⁻¹xy⁻¹xyx⁻¹y⁻¹x⁻¹x⁻¹yxxyxyx⁻¹x⁻¹y⁻¹`
    - `x⁻¹y⁻¹x⁻¹yx⁻¹y⁻¹x⁻¹yxxy⁻¹xyx`
    - `xyx⁻¹yxy⁻¹y⁻¹x⁻¹`
    - `y⁻¹xy⁻¹xy⁻¹x⁻¹x⁻¹x⁻¹yyyx⁻¹y⁻¹xxy`

### $n = 3, l = 15$
* 93 слова из $\langle x\rangle$ за 75 с
  
    Примеры:
    - `zx⁻¹x⁻¹x⁻¹x⁻¹x⁻¹z⁻¹xx`
    - `y⁻¹x⁻¹y`
    - `zzx⁻¹x⁻¹x⁻¹x⁻¹z⁻¹xz⁻¹y⁻¹xyx`
    - `y⁻¹x⁻¹x⁻¹yz⁻¹y⁻¹x⁻¹yzx⁻¹`
    - `yxyx⁻¹x⁻¹y⁻¹y⁻¹z⁻¹xxzx⁻¹x⁻¹`

* 10 слов из $\langle x\rangle \cap \langle y\rangle$ за 145 c

    Примеры: 

    - `xy⁻¹y⁻¹xyx⁻¹yx⁻¹`
    - `z⁻¹y⁻¹xyx⁻¹z`
    - `zyyxy⁻¹x⁻¹y⁻¹z⁻¹`
    - `yx⁻¹y⁻¹x`
    - `xxyx⁻¹y⁻¹x⁻¹`

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle xyz\rangle$

### $n = 4, l = 30$
* 10 слов из $\langle x\rangle$ за 404 с
  
    Примеры:
    - `z⁻¹pxy⁻¹xyp⁻¹zx`
    - `xxzxz⁻¹xxz⁻¹xzyx⁻¹y⁻¹p⁻¹x⁻¹pxxz⁻¹xzxpxp⁻¹`
    - `x⁻¹x⁻¹p⁻¹yxxy⁻¹x⁻¹x⁻¹px⁻¹`
    - `xyxpx⁻¹p⁻¹y⁻¹x⁻¹x⁻¹`
    - `y⁻¹px⁻¹p⁻¹y⁻¹z⁻¹x⁻¹zxpx⁻¹p⁻¹yx⁻¹x⁻¹y`

* 0 слов из $\langle x\rangle \cap \langle y\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle p\rangle \cap \langle xyzp\rangle$

### $n = 5, l = 50$
* 0 слов из $\langle x\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle$

* 0 слов из $\langle x\rangle \cap \langle y\rangle \cap \langle z\rangle \cap \langle p\rangle \cap \langle q\rangle \cap \langle xyzpq\rangle$