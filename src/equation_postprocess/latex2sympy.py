from sympy.parsing.latex import parse_latex
import pandas as pd
from fuzzywuzzy import fuzz

categories = ['Integral','Derivative', 'Limit','Sum','sin','cos','tan','arctan', 'F(x)']
eq_csv = "/home/fm-pc-lt-123/Personal/Freelancing/math_ocr/equation_dataset.csv"

def rank_equation(pred):
    df = pd.read_csv(eq_csv)
    try:
        eq = parse_latex(pred)
        for category in categories:
            if category in str(eq):
                print("Category:",category)
                df = df[df['Category']==category]
        print("parsed:",eq)
    except:
        print("Invalid latex code")
    
    print("Len:",len(df))
    tuples_list = ([(fuzz.token_set_ratio(pred,j),j) for j in df['Latex']])
    # print("Tuple_list:",tuples_list)
    similarity_score, fuzzy_match = map(list,zip(*tuples_list))
    results = pd.DataFrame({"fuzzy_match": fuzzy_match, "similarity_score":similarity_score})
    results.sort_values(by=["similarity_score"], axis=0, ascending=False, inplace=True)
    print(results)

if __name__ == "__main__":

    text_1 = r"\frac{d}{dx}(x^{2}+x)=-\frac{a}{c}+\frac{d^{2}y}{d t^{2}}+\frac"
    text_2 = r'x^{2}\frac{d^{2}y}{d x^{2}}=\frac{d^{2}y}{d t^{2}}-\frac{b}{c}'
    text_3 = r'x^{2}+y^{2}+4x-2y=-1'
    text_4 = r'\lim_{h \to 0 } \frac{f(x+h)-f(x)}{h}'
    text_5 = r"x\int_{a}^b f(x)dx"
    text_6 = r"\sum_{n=1}^{\infty} 2^{-n} = 1"
    text_7 = r"\arctg \frac{\pi}{3} = \sqrt{3}"
    # text_8 = r"\begin{bmatrix}a & b \\c & d \end{bmatrix}"

    eq1 = parse_latex(text_1)
    eq2 = parse_latex(text_2)
    eq3 = parse_latex(text_3)
    eq4 = parse_latex(text_4)
    eq5 = parse_latex(text_5)
    eq6 = parse_latex(text_6)
    eq7 = parse_latex(text_7)
    # eq8 = parse_latex(text_8)

    print(str(eq1),str(eq2),str(eq3),str(eq4),str(eq5), str(eq6), str(eq7))