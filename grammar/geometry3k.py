import math, operator, functools
import lark

class Parser(lark.Transformer):
  @lark.v_args(inline=True)
  def frac(self, x: float, y: float): return x/y
  @lark.v_args(inline=True)
  def sqrt(self, x: float): return math.sqrt(x)
  @lark.v_args(inline=True)
  def pi(self): return math.pi
  @lark.v_args(inline=True)
  def FLOAT(self, x: str): return float(x)
  @lark.v_args(inline=True)
  def div(self, x: float, y: float): return x/y
  @lark.v_args(inline=True)
  def add(self, x: float, y: float): return x+y
  @lark.v_args(inline=True)
  def sub(self, x: float, y: float): return x-y
  @lark.v_args(inline=True)
  def mul(self, x: float, y: float): return x*y
  def expr(self, X: list): return functools.reduce(operator.mul, X)
  @lark.v_args(inline=True)
  def start(self, x: float): return x

GRAMMAR = lark.Lark.open("geometry3k.lark", rel_to=__file__)
PARSER = Parser()

def compute(S: str) -> float:
  return PARSER.transform(GRAMMAR.parse(S))
