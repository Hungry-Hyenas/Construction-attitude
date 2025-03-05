from konlpy.tag import Kkma
from konlpy.utils import pprint

kkma = Kkma()
pprint(kkma.sentences(u'배수로 드레인 이동통로 확보상태 미흡, 작업자의 불안전한 상태에서의 이동 작업'))

pprint(kkma.nouns(u'배수로 드레인 이동통로 확보상태 미흡, 작업자의 불안전한 상태에서의 이동 작업'))