
function splitFormat(lines) {
  let map = new Map();
  let temp = '';
  let arr = [];


  for (let line of lines) {
      temp = line.split(', ');
      if (map.has(temp[1])) {
          arr = map.get(temp[1]);
          arr.push({
              date: new Date(temp[2]).getTime(),
              format: temp[0].split('.')[1]
          });
          map.set(temp[1], arr);
      } else {
          map.set(temp[1], [{
              date: new Date(temp[2]).getTime(),
              format: temp[0].split('.')[1]
          }]);
      }
  }

  for (let [key, val] of map.entries()) {
      val.sort((a, b) => (a.date > b.date) ? 1 : ((b.date > a.date) ? -1 : 0))
  }
  
  return map;
}

function getIndexNformat(map, city, time) {
  let arr = map.get(city);

  let index = arr.findIndex(x => x.date == new Date(time).getTime());

  let format = arr[index].format;

  let zero = '';
  index += 1;
  let count = Math.floor(Math.log10(arr.length)) - Math.floor(Math.log10(index));
  for (let i = 0; i < count; i++) {
      zero += '0';
  }

  return `${zero + index + '.' + format}`;
}

function solution(S) {
  let str = '';
  let lines = S.split('\n');
  let map = splitFormat(lines); // split newline
  let temp = '';
  let city = '';
  let time = '';

  for (let line of lines) {
      temp = line.split(', ');
      city = temp[1];
      time = temp[2];
      indexNformat = getIndexNformat(map, city, time);
      str += `${city + indexNformat + '\n'}`;
  }

  return str;
}



let answer = solution('photo.jpg, Warsaw, 2013-09-05 14:08:15\njohn.png, London, 2015-06-20 15:13:22\nmyFriends.png, Warsaw, 2013-09-05 14:07:13\nEiffel.jpg, Paris, 2015-07-23 08:03:02\npisatower.jpg, Paris, 2015-07-22 23:59:59\nBOB.jpg, London, 2015-08-05 00:02:03\nnotredame.png, Paris, 2015-09-01 12:00:00\nme.jpg, Warsaw, 2013-09-06 15:40:22\na.png, Warsaw, 2016-02-13 13:33:50\nb.jpg, Warsaw, 2016-01-02 15:12:22\nc.jpg, Warsaw, 2016-01-02 14:34:30\nd.jpg, Warsaw, 2016-01-02 15:15:01\ne.png, Warsaw, 2016-01-02 09:49:09\nf.png, Warsaw, 2016-01-02 10:55:32\ng.jpg, Warsaw, 2016-02-29 22:13:11');
console.log(answer);
