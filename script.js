let modelo;
let lossValues = [];

async function entrenarModelo() {
  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  modelo.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const ys = tf.tensor2d(xs.dataSync().map(x => x * -2 + 6), [9, 1]);

  lossValues = [];

  await modelo.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        lossValues.push({ x: epoch + 1, y: logs.loss });
      },
      onTrainEnd: () => {
        document.getElementById("train-status").innerHTML = 
          '<span style="color: green;">✅ Entrenamiento finalizado. ¡Modelo listo!</span>';
        graficarPerdida();
      }
    }
  });
}

async function hacerPrediccion() {
  const input = document.getElementById("inputX").value;
  const valores = input.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

  if (valores.length === 0) {
    document.getElementById("result").innerText = "Por favor, ingresa uno o más números válidos separados por comas.";
    return;
  }

  const inputTensor = tf.tensor2d(valores, [valores.length, 1]);
  const predicciones = await modelo.predict(inputTensor).data();

  // Ordenar por valor de X para mejor presentación
  const resultados = valores.map((x, i) => ({ x, y: predicciones[i] }));
  resultados.sort((a, b) => a.x - b.x);

  let resultadoHTML = "<strong>Resultados:</strong><br>";
  resultados.forEach(r => {
    resultadoHTML += `Para x = ${r.x} ➜ y = ${r.y.toFixed(2)}<br>`;
  });

  document.getElementById("result").innerHTML = resultadoHTML;
}

function graficarPerdida() {
  const ctx = document.getElementById("lossChart").getContext("2d");

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Pérdida (Loss)',
        data: lossValues,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
        fill: false
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Época'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Valor de pérdida'
          },
          suggestedMax: 1,
          beginAtZero: true
        }
      }
    }
  });

  const perdidaInicial = lossValues[0].y;
  const perdidaFinal = lossValues.at(-1).y;
  const reduccion = 100 * (1 - perdidaFinal / perdidaInicial);

  document.getElementById("loss-info").innerText =
    `Pérdida inicial: ${perdidaInicial.toFixed(4)}, ` +
    `Pérdida final: ${perdidaFinal.toFixed(4)} ` +
    `(Reducción: ${reduccion.toFixed(2)}%)`;
}
