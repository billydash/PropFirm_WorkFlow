import numpy as np
import pandas as pd

columns=["eleanour", "tahani", "jason", "chidi"]
my_data=np.random.randint(0 , 101,size=(3, 4))
children = pd.DataFrame(data=my_data, columns=columns)
print(children)
print(children.iloc[0, 0])
children["janet"]=children["tahani"]+children["jason"]