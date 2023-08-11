<template>
    <div class="app-container">
        <img src="/sparkify_logo_v1.png" alt="Sparkify Logo" class="app-logo">
        <h1 class="app-subtitle">Churn Analysis tool for Sparkify</h1>
        <div class="app-description">
            <p> Dive deep into user behavior of Sparkify's music streaming service. </p>
            <p> <strong>Gain clear insights into churn patterns.</strong></p>
            <p> <em>Upload your dataset and watch as our tool generates insights within moments.</em> </p>
        </div>
        <div class="file-upload-section">
            <input type="file" @change="handleFileUpload" class="file-input" />
        </div>
        <div class="chart-container">
            <canvas id="myChart" class="chart" width="800" height="400"></canvas>
            <canvas v-if="state.selectedUserId" id="contribChart" class="chart" width="800" height="600"></canvas>
        </div>
    </div>
</template>


<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

body {
    background-color: #f5f7f9;
    color: #333;
}

.app-container {
    font-family: 'Lato', sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    text-align: center;
}

.app-logo {
    max-width: 300px;
    margin-bottom: 20px;
}


.app-title {
    font-size: 40px;
    color: #2d3748;
    font-weight: 700;
}

.app-subtitle {
    font-size: 24px;
    color: #4a5568;
    margin-bottom: 20px;
}

.app-description p {
    font-size: 18px;
    line-height: 1.6;
}

.file-upload-section {
    margin: 20px 0;
}

.file-label {
    font-weight: 700;
    display: block;
    margin-bottom: 5px;
}

.file-input {
    padding: 10px;
    cursor: pointer;
    color: #2d3748;
    /* background-color: #f5f7f9; */
    border: none;
    border-radius: 5px;
    /* transition: background-color 0.3s ease; */
}

/* 
.file-input:hover {
    background-color: #e2e8f0;
} */

.chart-container {
    margin-top: 30px;
}

.chart {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}
</style>

  
  
<script setup lang="ts">
import { nextTick, reactive } from 'vue';
import { Chart, registerables, ChartTypeRegistry, Point, BubbleDataPoint } from 'chart.js';


Chart.register(...registerables);

const state = reactive({
    myChart: null as Chart<keyof ChartTypeRegistry, (number | Point | [number, number] | BubbleDataPoint | null)[], unknown> | null,
    contribChart: null as Chart<keyof ChartTypeRegistry, (number | Point | [number, number] | BubbleDataPoint | null)[], unknown> | null,
    selectedUserId: null as string | null,
    responseData: { prediction: [] as any[] },
    chartUpdating: false,
});

const handleFileUpload = async (e: Event) => {
    const inputElement = e.target as HTMLInputElement;
    if (!inputElement || !inputElement.files || inputElement.files.length === 0) {
        return;
    }

    const file = inputElement.files[0];
    const formData = new FormData();
    formData.append('user_data_file', file);

    const apiUrl = import.meta.env.VITE_API_URL;
    const apiResponse = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        body: formData,
    });

    state.responseData = await apiResponse.json();

    const labels: string[] = [];
    const datasetData: number[] = [];

    state.responseData?.prediction.forEach((item: any) => {
        labels.push(item.userId);
        datasetData.push(item.churn_probability);
    });

    nextTick(() => {
        createChart('myChart', 'bar', labels, datasetData, 'Churn Probability', handleBarClick, null, null);
    });
};

const handleBarClick = (userId: string) => {
    state.selectedUserId = userId;

    nextTick(() => {
        createContribChart();
    });
};

const createContribChart = () => {
    const user = state.responseData?.prediction.find((item: any) => item.userId === state.selectedUserId);

    const contribLabels: string[] = [];
    const contribData: number[] = [];
    const contribColors: string[] = [];

    for (let key in user.contributions) {
        contribLabels.push(key);
        contribData.push(user.contributions[key]);
        contribColors.push(user.contributions[key] >= 0 ? 'green' : 'red');
    }

    nextTick(() => {
        createChart(
            'contribChart',
            'H',
            contribLabels,
            contribData,
            'Contributions',
            null,
            contribColors,
            user.userId
        );
    });
};

const createChart = (
    canvasId: string,
    chartType: string,
    labels: string[],
    datasetData: number[],
    label: string,
    onClick: any,
    colors: string[] | null,
    userId: string | null
) => {
    const ctx = document.getElementById(canvasId) as HTMLCanvasElement;

    if (!ctx) {
        console.error(`Canvas element with id ${canvasId} not found in DOM.`);
        return;
    }

    let chart: any | null = state.myChart;
    if (canvasId === 'contribChart') {
        chart = state.contribChart;
    }

    nextTick(async () => {
        if (chart) {
            chart.destroy(); // destroy previous chart if it exists
        }

        // Delay chart creation to ensure canvas is properly initialized
        await new Promise((resolve) => setTimeout(resolve, 100));

        const indexAxis = chartType === 'H' ? 'y' : 'x';

        state.chartUpdating = true; // set flag to true

        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label,
                        data: datasetData,
                        backgroundColor: colors as string[] | undefined,
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                indexAxis,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            autoSkip: false,
                        },
                    },
                    x: {
                        beginAtZero: true,
                        ticks: {
                            autoSkip: false,
                        },
                    },
                },
                onClick: (e) => {
                    if (!onClick || state.chartUpdating || !chart) {
                        // check flag before updating chart
                        return;
                    }

                    state.chartUpdating = true; // set flag to true
                    const canvas = chart.canvas;
                    if (!canvas) {
                        console.error('Canvas element not found.');
                        state.chartUpdating = false; // set flag to false
                        return;
                    }
                    const ctx = canvas.getContext('2d');
                    if (!ctx) {
                        console.error('Canvas context not available.');
                        state.chartUpdating = false; // set flag to false
                        return;
                    }

                    const elements = chart.getElementsAtEventForMode(e,
                        'nearest',
                        { intersect: true },
                        false
                    );
                    if (elements.length && chart.data.labels) {
                        const index = elements[0].index;
                        onClick(chart.data.labels[index]);
                    }
                    state.chartUpdating = false; // set flag to false
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            generateLabels: (chart) => {
                                const originalGenerateLabels = Chart.defaults.plugins.legend.labels.generateLabels;
                                const defaultLabels = originalGenerateLabels(chart);

                                if (chart.canvas.id === 'contribChart') {
                                    defaultLabels.forEach((label) => {
                                        label.fillStyle = '#000'; // Set legend label color to black
                                        label.text = `UserId: (${userId})`; // Set the legend label to "UserId: (userId)"
                                    });
                                }

                                return defaultLabels;
                            },
                        },
                    },
                },
            },
        });

        if (canvasId === 'myChart') {
            state.myChart = chart;
        } else if (canvasId === 'contribChart') {
            state.contribChart = chart;
        }

        state.chartUpdating = false; // set flag to false
    });
};
</script>
  