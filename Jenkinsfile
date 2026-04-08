pipeline {
    agent any

    environment {
        IMAGE_NAME = "pneumonia-app"
        CONTAINER_NAME = "pneumonia-container"
    }

    stages {

        stage('Clone Repo') {
            steps {
                git 'https://github.com/Stark-017/Deep-Learning.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t %IMAGE_NAME% .'
            }
        }

        stage('Run Container') {
            steps {
                bat '''
                docker stop %CONTAINER_NAME% || true
                docker rm %CONTAINER_NAME% || true
                docker run -d -p 5000:5000 --name %CONTAINER_NAME% %IMAGE_NAME%
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Deployment Successful!'
        }
        failure {
            echo '❌ Build Failed!'
        }
    }
}
