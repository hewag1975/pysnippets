
## lists
## https://docs.python.org/3/tutorial/datastructures.html

## create list
dict = ["x1", "x2", "x3", "y4"]
dict

## print ith list element
for d in dict:
    print(d)

## capitalize 
[d.capitalize() for d in dict]

## append
dict_app = "z5"
dict.app(dict_append)
dict

## reverse
dict_rev = dict.reverse()

## indices
dict.index("y4")

## list comprehension
squares = []
for x in range(10):
    squares.append(x**2)
squares

### equivalent to
squares = map(lambda x: x**2, range(10))
squares = list(squares)

squares = [x**2 for x in range(10)]

## replace every 'x' with an 'a'
dictnew = [d.replace("x", "a") for d in dict]
dictnew


## create dictionary
