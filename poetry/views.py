from django.shortcuts import render
from utils.poetryGenerator import couplet, quatrain, sonnet

def index(request):
    poemType = request.POST.get('poemType')
    poem = []
    if (poemType == 'couplet'):
        poem = couplet()
    elif (poemType == 'quatrain'):
        poem = quatrain()
    elif (poemType == 'sonnet'):
        poem = sonnet()
    return render(request, 'poetry/index.html', {'poem': poem})
