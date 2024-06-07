import json
import requests
import csv
import requests
from bs4 import BeautifulSoup
import os

htmlFilePath = r'/Users/partha/wine.html'
url = "https://www.wine-searcher.com/find/austin+hope+cab+sauv+paso+robles+st+luis+obispo+county+central+coast+california+usa/1/usa#t2"

class Vivino:
    def get_tastes(self,wine_id):
        url = 'https://www.vivino.com/api/wines/'+ str(wine_id) +'/tastes?language=en'
        headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.vivino.com/explore?e=eJzLLbI11rNQy83MszU0UMtNrLA1MVBLrrQNDVZLBhIuagW2hmrpabZliUWZqSWJOWr5RSm2avlJlbZq5SXRsUBJMGUEoYwhlIlasa2zIwARKxyB',
        'X-Requested-With': 'XMLHttpRequest',
        'Content-Type': 'application/json',
        'DNT': '1',
        'Alt-Used': 'www.vivino.com',
        'Connection': 'keep-alive',
        'Cookie': 'first_time_visit=w4cUGBwilRNx%2B%2F%2FWdhKSdb7xX%2BFKsiWdTVkYpSP%2B7QoX3w4jJ2zHCwpn2aUfbvThgJ8Od2UglGpiQwqMkihleryUp0fJCgTjOyRuLDn5pvxvUffX2o9r3FuSOAwgGtk0FX4%3D--LXG5zV4CcZ3kW8%2Fa--yEc7HAdI8fe39YezY1q8%2BA%3D%3D; anonymous_tracking_id=aQUCdyYUx4POq9j%2B8B8Atk019axYN71lliKjkN9%2FMNPy1L49b85Rp8KMZL8RRYE3d7YAkYIKgH%2BIMlOEMSBUo%2BxNyTnaVYa6MLRKY500OSAVmFOL2jyQQ3%2FiWjaUPMUbeEh5sc%2FmKnmS9BOaL5FajoGJ%2FCB95iZdwur%2BFwaXYYRG55Z%2FE314EP%2BUj1UdDlDk6vtC--NdG6D9qss0w%2B4F0N--PRweDR3rywVY3I9yevrTUQ%3D%3D; _ruby-web_session=n6CPauHN1DSzYB3wEifnP70aBY0Md94aOJvSaQRL9sEBiaxQlBZUTRovLNfoEbyfHRpiiJFhLUoqVUwCzI4jpwXxvLBfwNgCHx6N00v4Rkwx0H%2FwVncNCqIiS1WFakUg38PIXlMauPwiWJHpdLvkgLmSE95BQTv5YClP%2FcbMEP%2Fr3wWin3cjgvWu%2Fn%2B2T%2BHszSZzzXEM1CnMdipyTQXbf3E1BQsOyVz9XPygVcLUdj1FkwBnNGqNa6AU6topfJnyspEV9QAM9S84LjQseI07noMIH5%2FuHAxpuUCeRcx1Ck8MwAdumWBSEOjn0x88V1oBAbtsnykYgSdQGmWlZ3wcWy08oRDlR5WAxsnFp80MncZ8l02THPZT4GMjzHcPX4ta9uOsn9EIAP2URJ0YAPqYFBH3KqvNeu0bpkPZi3JvbDa58cTfGciVog5mz%2BbAGfSpH5oaaff5nXhlAge1P742CF0x3wahNaplmR27IneqN5pgZ5rtUciZ7qXXr1d4Nty9J%2FhopMZ59w%3D%3D--8TXG9inPdpd0vIeV--BdQrQ0k8pq3rbUOFquRTIQ%3D%3D; eeny_meeny_personalized_upsell_module_v2=PwbytU8qMh7%2FTbv3J1vaX0vMG0r0nNEyDLJPkmGhGQceUwj81V%2B2MNXC3wBCAJssB51vj2k9C2c0yKPGNyjiMw%3D%3D; recently_viewed=NPveT5bDRAHm1roMVV%2F7LFayPFn1a8dX3cMQcc%2BWLxIix2STR%2BALwtG%2BzR9Xgk767UvmnpaArxSH%2BG2%2BPWiCG5N2vmJQghZERqJQMszQteEiDxG13%2FaTCW8S2gQfoWa5IA6v3N5X%2ByaHLNEw02Rm34cFbBgByNYm2Y15uGxyilfq5%2BmAQScr6R%2BsFhFyOj31Rw5%2ByWbz5xVuRTRN1iaIgQ5t9q8%2FMcAvHQgIUzyDZ4zpeY6iSEHD5cJei6YX0HJG0XJB2lJ4NYLib9CPcqTSbquqL65ww%2BMwErSjClNzSn6WEMueSL8hTd3NAyOWOjpKLx3vDPe2m1zFlIowg7DgnKmt8bXx9OEh6%2FMMVzAv9N466JUABY3fmMdFtmpbOnC7t%2FIvQCisiV4pqcWMBsfJaOutDnMjMmg8dqczIHUb%2FL2%2BFQLQ7vd0mbgb3vqaq%2F%2BSq78ILbgKlD0WAiBpR26TZKzOP0RJBSrweIE%2F7MGaXErlPunPxLXUIDFIz43OOGtjpbO2aVQzXaBeJicbzzekPWw%3D--uEfC7023J2K7AJsx--twcLSZa%2F6dJAKDdhlp%2FBUw%3D%3D; csrf_token=TsjnBkkWwaKyUGXCG6QpaHs7MK4TQkKBgr83P-XFXZKJTYdx3648e1BUD1ihDoMZKRPDk1quD05rw-wtI1xWKQ; client_cache_key=jPJWubP2E%2BeUyfdtd1807jOYi6qK2aHgjxWU30Zg7%2B9353RheVVOZvobwlGzpGd3MyC04bwVx1huAdwaRif%2FI%2FQu4BD3W%2BYv8d7iaIlkPEznjqHPOmuxa%2BfaZPikzkT96R1UyGbXMZ%2BRbb21%2B3SjEtOB--pD%2B4u9RdxDctCQar--3ugpx7Pg2YdaUForMjr05g%3D%3D; _ruby-web_session=G52Jj9VffD1zk%2FvfKnRcWqTgRf%2FR4Ab%2F18ePnZ16R0YBXvtJpPyAfi%2FLYSMrYLaL3%2Fr%2FpsL6bIT35CwWPjYaiR3rGz9DL3x4ko%2FfngrBeT3VxePd7K8wke%2Fk6VwyP5C%2Fk%2Bl3arO0A61GtUtnQZboZHnsae5i%2FRdCc06fy6HfUjcU4ULjsXuXu9aQv0wdw3f0vVI3ZEblhlicBlJ0BF%2FG5JvgaKFS%2B8BGlpoICu3HKpdgvX2DQE%2FdAv7JR8AwlfRmAiKFp7%2BgfnhJiPxv%2BkH7AqGWxwgxeFLszYOCZDz1soeqg%2BChO%2BwK%2BlbffgTlQpi9eCvdfZfozYTt9Cr9wsVvlvMGSr1dWOyjIuMCopP21u%2FTlZKKeoPZckyRYfTP%2F2SdjxmZQEKSJ%2BLmhEqbdsDW2mEZ7HK8f4Yd6EIUEwav6wXJTvNVzlFo1v8efkKDOiuREKsNKbRoEJoaifnszWcp2NXDjGywzDIuR7wGq8SdPWLfv6TFdFpsSbRDk3%2FOQWOPPzRI%2BYtKfixCsDeGnXFr8QP6p99U6Ge0rNEcBdw6tA%3D%3D--Fq4ttv4gpyRB0BIv--9iqj1wYp0gTkD4OlGjypXA%3D%3D; csrf_token=lL3kz115wdo9lBerG9XWxBOsExiQyUdQKPg0QkWDOVtTOIS4y8E8A9-QfTGhf3y1QYTgJdklCp_BhO9Qgxoy4A; eeny_meeny_personalized_upsell_module_v2=s0fDEL5EEJlMXIrJuSa5Nbxf9BiRPipalTX8PxulxbyZHFp5Qc7G%2BhCRuGLYbgcLUq4tmJV3WocjHIUIHIshug%3D%3D',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'TE': 'trailers'
        }

        response = requests.request("GET", url, headers=headers, data={})
        if response.status_code == 200:
            data = json.loads(response.text)
            return data['tastes']['flavor']
        else:
            return None
        
    def get_vivino_data(self):
        with open(htmlFilePath, 'r') as file:
            for line in file:
                if 'window.__PRELOADED_STATE__.winePageInformation' in line:
                    json_str = line.split(' = ')[1].strip()[:-1]
                    data = json.loads(json_str)
                    break

        if(data):
            name = data['vintage']['name']
            price = data['highlights'][0]['metadata']['price']['amount']
            rating = data['vintage']['statistics']['ratings_average']
            style = data['wine']['style']['name']
            tastes_structure = data['wine']['style']['baseline_structure']
            foods = data['wine']['style']['food']  
            wine_id = data['vintage']['wine']['id']
            flavours = self.get_tastes(wine_id)

            data = {
                'serial':'',
                'name':name,
                'price':price,
                'rating':rating,
            }
            for taste in tastes_structure:
                data[taste] = tastes_structure[taste]
                # df[taste] = [tastes_structure[taste]]
            data['Foods'] = ','.join([food['name'] for food in foods])
            for flavor in flavours:
                prim_kw = flavor['primary_keywords']
                mention_count = flavor['stats']['mentions_count']
                kw_text = ','.join([kw['name'] for kw in prim_kw])
                data[flavor['group']+ ' Notes'] = str(mention_count) + '|' + kw_text

            return data
        

class WineSearcher:
    def get_winesearcher_data():
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0'
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('h1').text.strip()
        style = soup.select('div.font-light-bold.mt-1.mt-md-0.text-gray-900')[1].text.strip()
        location = soup.find('span', class_='align-middle smaller font-light-bold').text.strip()
        pair = soup.select('div.font-light-bold.mt-1.mt-md-0.text-gray-900')[3].text.strip()

        data = {'title':title,'style':style,'location':location,'pair':pair}
        return data

wsData = WineSearcher.get_winesearcher_data()
vivinoData = Vivino().get_vivino_data()

data = vivinoData
data['style'] = wsData['style']
data['location'] = wsData['location']
data['foodParing'] = wsData['pair']

root, ext = os.path.splitext(htmlFilePath)
outputFilePath = os.path.join(root + '.csv') 

with open(outputFilePath, 'w') as file:
     writer = csv.DictWriter(file, fieldnames=data.keys())
     writer.writeheader()
     writer.writerow(data)