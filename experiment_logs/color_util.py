
def rgb_to_hex(rr, gg, bb):
    rgb = (rr, gg, bb)
    return '#%02x%02x%02x' % rgb

map_cc = {  'bb': ['BBA', '#3E85BA', '--', 'x'], 
            'robustmpc': ['RobustMPC', rgb_to_hex(122, 122, 122), '-.', 'o'], 
            'pensiedtp':['Pitree(P)', '#00C1D4', '-.', 'v'], 
            'newtree': ['Pitree_change', '#FFA0A0', '--', '^'],
            'ComTreep': ['ComTree(P)', 	'#FF0000' ,':', '>'], #C84B31',
            'rl': ['Pensieve', '#FFA0A0', ':', 'D'],
           'log_tree_': ['ComTree-L', 	rgb_to_hex(122, 122, 122) ,':', '>'], #C84B31',
            'log_opt_': ['ComTree(P)', '#FF0000', '--', 'H'],
            'log_opta_': ['ComTree(P)-L','#FF8000', '-.', 'h'],
            'ghent': ['Genet', '#FF8000', ':', '>'], 
                'genet': ['Genet', '#FF8000', ':', '>'], 
            'bola': ['Bola', '#247881', ':', '1'],
            'pensieve': ['Pensieve', '#FFA0A0', ':', 'D'],
            'pitreep':['Pitree(P)', '#00C1D4', '-.', 'v'], 
            'BB': ['BBA', '#3E85BA', '--', 'x'], 
            'robustMPC': ['RobustMPC', rgb_to_hex(122, 122, 122), '-.', 'o'], 
            'log_pitreea': ['Pitree(P)-L',  '#3E85BA', '-', 'x'], 
             'llm': ['NetLLM', '#247881', ':', '1'],
               'log_treea_': ['ComTree_C-L', '#247881', ':', '1'],
                    'log_pensiedt_':['Pitree(P)', '#00C1D4', '-.', 'v'], 
          
        }